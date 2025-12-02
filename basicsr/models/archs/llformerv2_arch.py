import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange

# ------------------- Layer Normalization ------------------- #
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.layer_norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

# ------------------- Overlapped image patch embedding with 3x3 Conv ------------------- #
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

# ------------------- Resizing modules ------------------- #
class Unet_Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Unet_Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

# ------------------- Axis-based Multi-head Self-Attention ------------------- #
class NextAttentionImplZ(nn.Module):
    def __init__(self, dim, num_head, bias):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.q1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.q3 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        num_head, dim_head = self.num_head, c // self.num_head

        qkv = self.q3(self.q2(self.q1(x)))
        reshape = lambda x: einops.rearrange(x, "b (nh dh) h w -> (b nh h) w dh", nh=num_head, dh=dim_head)
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.fac
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "(b nh h) w dh -> b (nh dh) h w", nh=num_head, dh=dim_head, b=b, h=h)
        out = self.fin(out)

        return out

# ------------------- Row and Col Attention ------------------- #
class NextAttentionZ(nn.Module):
    def __init__(self, dim, num_head, bias=True):
        super().__init__()
        assert dim % num_head == 0
        self.num_dims = dim
        self.num_head = num_head
        self.row_att = NextAttentionImplZ(dim, num_head, bias)
        self.col_att = NextAttentionImplZ(dim, num_head, bias)

    def forward(self, x):
        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)
        return x

# ------------------- Dual Gated Feed-Forward Network ------------------- #
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, illu=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.illu = illu
        if self.illu:
            self.illu_conv = nn.Conv2d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)

    def forward(self, input):
        if self.illu:
            x, illu = input
        else:
            x = input

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        if self.illu:
            illu1, illu2 = self.illu_conv(illu).chunk(2, dim=1)
            mask = x1 * F.sigmoid(illu1) + x2 * F.sigmoid(illu2)
            x = F.gelu(x2) * x1 + F.gelu(x1) * x2
            x = mask * x + x
        else:
            x = F.gelu(x2)*x1 + F.gelu(x1)*x2

        x = self.project_out(x)
        return x

# ------------------- Depthwise Convolution ------------------- #
class DepthwiseConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 5, padding=5//2, groups=dim),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return self.conv(x)

# ------------------- Illumination Map Estimation ------------------- #
class Illumination_Map_Estimation(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        self.projs = nn.ModuleList()
        for level in range(1,4):
            n_feat = int(dim * (2**level) * ffn_expansion_factor)
            self.projs.append(nn.Conv2d(5, n_feat, 3, 1, 1))

    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        prior = torch.cat((x, x_max, x_mean), dim=1)

        illu_map = []
        illu = prior
        for level in range(1,4):
            factor = 2**level
            illu = F.max_pool2d(prior, kernel_size=factor, stride=factor)
            illu_map.append(self.projs[level-1](illu))

        return illu_map

# ------------------- Axis-based Transformer Block V2 ------------------- #
class ATBV2(nn.Module):
    def __init__(self, dim=16, num_heads=1, ffn_expansion_factor=2.66, bias=False, num_slots=64, illu=False):
        super().__init__()
        self.pool_size = math.isqrt(num_slots)
        assert self.pool_size**2 == num_slots, ValueError('num_slots must be a perfect square')

        self.dim = dim
        self.illu = illu
        self.slot_scale = nn.Parameter(torch.tensor(dim ** (-0.5)))
        self.distribute_scale = nn.Parameter(torch.tensor(dim ** (-0.5)))

        self.norm1 = LayerNorm(dim)
        self.feature_conv = DepthwiseConvolution(dim)
        self.attention = NextAttentionZ(dim, num_heads)

        # 2nd distribution attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # ffn
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, illu)

        if illu:
            n_feat = int(dim * ffn_expansion_factor)
            self.attn_conv = nn.Conv2d(n_feat, dim, kernel_size=1, bias=bias)
            self.illu_conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, groups=n_feat, bias=bias)

    def _attention(self, input):
        if self.illu:
            x, init_slots, latent_feature = input
            prior_mask = F.sigmoid(latent_feature)
        else:
            x, init_slots = input

        b, c, h, w = x.shape
        init_slots = rearrange(init_slots, 'b c h w -> b (h w) c')
        x_flatten = rearrange(x, 'b c h w -> b (h w) c')

        cross_attn = torch.matmul(F.normalize(init_slots, dim=-1), F.normalize(x_flatten, dim=-1).transpose(-1, -2)) * self.slot_scale
        slots = torch.matmul(torch.softmax(cross_attn, dim=-1), x_flatten)

        slots = rearrange(slots, 'b (h w) c -> b c h w', h=self.pool_size, w=self.pool_size)
        slots = self.attention(slots)
        slots = rearrange(slots, 'b c h w -> b (h w) c')

        # ----- distribution attention -----
        q = self.to_q(x_flatten)
        k = self.to_k(slots)
        v = self.to_v(slots)
        distribute_attn = torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2)) * self.distribute_scale
        out = torch.matmul(torch.softmax(distribute_attn, dim=-1), v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        if self.illu:
            out = out * prior_mask + self.feature_conv(x) * (1.0 - prior_mask)

        return out

    def forward(self, input):
        if self.illu:
            x, illu = input
            illu_feature = self.illu_conv(illu)
            latent_feature = self.attn_conv(illu_feature)

            init_slots = F.adaptive_avg_pool2d(x, output_size=self.pool_size)
            x = x + self._attention((self.norm1(x), init_slots, latent_feature))
            x = x + self.ffn((self.norm2(x), illu_feature))
            return x, illu
        else:
            x = input
            init_slots = F.adaptive_avg_pool2d(x, output_size=self.pool_size)
            x = x + self._attention((self.norm1(x), init_slots))
            x = x + self.ffn(self.norm2(x))
            return x

##########################################################################
class LLFormerv2(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 16,
        num_enc_blocks = [4,4,6],
        num_dec_blocks = [4,4,6],
        num_bottleneck_blocks = 6,
        num_extraction_blocks = 6,
        num_latent_blocks = 2,
        num_refinement_blocks = 6,
        heads = [1,2,4,8,8],
        num_slots = [64,64,64,64,64],
        ffn_expansion_factor = 2.66,
        bias = False,
        shortcut = False
    ):
        super().__init__()
        self.shortcut = shortcut
        self.illu_map = Illumination_Map_Estimation(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor)
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, dim=int(dim))

        self.feature_extraction = nn.Sequential(*[ATBV2(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, num_slots=num_slots[0]) for _ in range(num_extraction_blocks)])

        self.latent = nn.Sequential(*[ATBV2(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, num_slots=num_slots[0]) for _ in range(num_latent_blocks)])

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.coefficients = nn.ParameterList()

        num_feat = int(dim)
        for level, num_block in enumerate(num_enc_blocks):
            self.downsamples.append(Unet_Downsample(num_feat))
            num_feat = num_feat * 2
            self.encoders.append(nn.Sequential(*[ATBV2(dim=num_feat, num_heads=heads[level+1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, num_slots=num_slots[level+1], illu=True) for _ in range(num_block)]))

        self.downsamples.append(Unet_Downsample(num_feat))
        num_feat = num_feat * 2
        self.bottleneck = nn.Sequential(*[ATBV2(dim=num_feat, num_heads=heads[-1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, num_slots=num_slots[-1]) for _ in range(num_bottleneck_blocks)])

        for level, num_block in enumerate(num_dec_blocks):
            self.upsamples.append(Unet_Upsample(num_feat))
            num_feat = num_feat // 2
            self.coefficients.append(nn.Parameter(torch.ones((2, num_feat))))
            self.skips.append(nn.Conv2d(num_feat, num_feat, kernel_size=1, bias=bias))
            self.decoders.append(nn.Sequential(*[ATBV2(dim=num_feat, num_heads=heads[-(level+2)], ffn_expansion_factor=ffn_expansion_factor, bias=bias, num_slots=num_slots[-(level+2)]) for _ in range(num_block)]))

        self.upsamples.append(Unet_Upsample(num_feat))
        num_feat = num_feat // 2
        self.coefficients.append(nn.Parameter(torch.ones((2, num_feat))))
        self.refinement = nn.Sequential(*[ATBV2(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, num_slots=num_slots[0]) for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):
        illu_map = self.illu_map(inp_img)
        inp_enc_extration = self.patch_embed(inp_img)

        out_enc = self.feature_extraction(inp_enc_extration)
        out_latent = self.latent(out_enc)

        encoder_feature = []
        for encoder, down, illu in zip(self.encoders, self.downsamples[:-1], illu_map):
            inp_enc = down(out_enc)
            out_enc, _ = encoder((inp_enc, illu))
            encoder_feature.append(out_enc)

        inp_bottleneck = self.downsamples[-1](out_enc)
        out_dec = self.bottleneck(inp_bottleneck)

        for decoder, up, skip, coeff in zip(self.decoders, self.upsamples[:-1], self.skips, self.coefficients[:-1]):
            inp_dec = up(out_dec)
            inp_dec = coeff[0, :][None, :, None, None] * inp_dec + coeff[1, :][None, :, None, None] * encoder_feature.pop()
            inp_dec = skip(inp_dec)
            out_dec = decoder(inp_dec)

        inp_refinement = self.upsamples[-1](out_dec)
        inp_refinement = self.coefficients[-1][0, :][None, :, None, None] * inp_refinement + self.coefficients[-1][1, :][None, :, None, None] * out_latent
        out_refinement = self.refinement(inp_refinement)

        if self.shortcut:
            out = self.output(out_refinement) + inp_img
        else:
            out = self.output(out_refinement)

        return out

if __name__ == '__main__':
    import time

    device = 'cuda:1'

    net = LLFormerv2(
        inp_channels=3,
        out_channels=3,
        dim = 16,
        num_enc_blocks = [2,4,4],
        num_dec_blocks = [4,4,2],
        num_bottleneck_blocks = 2,
        num_extraction_blocks = 1,
        num_latent_blocks = 1,
        num_refinement_blocks = 2,
        heads = [1,2,4,8,8],
        num_slots = [64,64,64,64,64],
        ffn_expansion_factor = 2.66,
        bias = False,
        shortcut = False,
    ).to(device)

    # x = torch.randn(1, 3, 400, 608).to(device)
    # x = torch.randn(1, 3, 2160, 3840).to(device)

    repeat = 100
    x = torch.randn(1, 3, 256, 256).to(device)
    start_time = time.time()
    for _ in range(repeat):
        with torch.no_grad():
            y = net(x)
    elapsed_time = time.time() - start_time
    print(f'Inference time: {elapsed_time / repeat:.4f} seconds')

    # net_cls_str = f'{net.__class__.__name__}'
    # net_params = sum(map(lambda x: x.numel(), net.parameters()))
    # print(f'Network: {net_cls_str}, with parameters: {net_params:,d}')

    # net_cls_str = f'{net.__class__.__name__}'
    # total_params = 0
    # for name, child in net.named_children():
    #     num_params = sum(p.numel() for p in child.parameters())
    #     print(f'{name}: {num_params:,d} parameters')
    #     total_params += num_params
    # print(f'Network: {net_cls_str}, with parameters: {total_params:,d}')

    # from thop import profile, clever_format
    # x = torch.randn(1, 3, 256, 256).to(device)
    # macs, params = profile(net, inputs=(x,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(f'MACs: {macs}, Parameters: {params}')
