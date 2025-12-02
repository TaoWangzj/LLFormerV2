import importlib
import copy
from collections import OrderedDict
from os import path as osp
from copy import deepcopy
from tqdm import tqdm

import cv2
import torch
import torchvision.utils as tvu


from basicsr.models.archs import define_network
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.models.base_model import BaseModel

import pyiqa

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class FeMaSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = define_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # PyTorchIQA metrics
        # if self.opt['val'].get('metrics') is not None:
        #     self.metric_funcs = {}
        #     for _, opt in self.opt['val']['metrics'].items():
        #         mopt = opt.copy()
        #         name = mopt.pop('type', None)
        #         mopt.pop('better', None)
        #         self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)

        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

        self.net_g_best = copy.deepcopy(self.net_g)
        self.print_network(self.net_g)


    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_g.train()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_opt'):
            pixel_type = train_opt['fft_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_fft = cri_pix_cls(**train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)


    def optimize_parameters(self, current_iter, tb_logger):
        train_opt = self.opt['train']

        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        # fft loss
        if train_opt.get('fft_opt', None):
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_freq'] = l_fft

        l_total.mean().backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def padding(self):
        b, c, h, w = self.gt.size()
        padding_factor = self.opt['val']['padding_factor']

        # calculate padding size
        padding_h = ((h + padding_factor - 1) // padding_factor) * padding_factor
        padding_w = ((w + padding_factor - 1) // padding_factor) * padding_factor

        lq = self.lq.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        padded_image = cv2.copyMakeBorder(
            lq,
            top=0, bottom=padding_h - h,
            left=0, right=padding_w - w,
            borderType=cv2.BORDER_REFLECT
        )

        self.lq = torch.tensor(padded_image.transpose(2, 0, 1)).unsqueeze(0).to(self.lq.device)


    def padding_inverse(self):
        b, c, h, w = self.gt.size()
        _, _, padded_h, padded_w = self.lq.size()

        if h > padded_h or w > padded_w:
            raise ValueError('The size of the image is larger than the padded image.')

        self.output = self.output[:, :, :h, :w]

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        # restoration = self.net_g(self.lq)
        _, _, h, w = lq_input.shape

        if h * w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()

    def validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=True, use_image=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
            rgb2bgr (bool): Whether to save images using rgb2bgr. Default: True
            use_image (bool): Whether to use saved images to compute metrics (PSNR, SSIM), if not, then use data directly from network' output. Default: True
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img=False, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        dataset_name = dataloader.dataset.opt['name']
        save_img = False
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            # padding to multiple of 16 (specially for LLformer)
            if self.opt['val'].get('padding', False):
                self.padding()

            self.test()

            if self.opt['val'].get('padding', False):
                self.padding_inverse()

            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # # calculate metrics
                # for name, opt_ in self.opt['val']['metrics'].items():
                #     tmp_result = self.metric_funcs[name](*metric_data)
                #     self.metric_results[name] += tmp_result.item()

                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(metric_data[0], metric_data[1], **opt_)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.is_train:
                if self.key_metric is not None:
                    # If the best metric is updated, update and save best model
                    to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                                self.metric_results[self.key_metric], current_iter)

                    if to_update:
                        for name, opt_ in self.opt['val']['metrics'].items():
                            self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                        self.copy_model(self.net_g, self.net_g_best)
                        self.save_network(self.net_g, 'net_g_best', current_iter)
                else:
                    # update each metric separately
                    updated = []
                    for name, opt_ in self.opt['val']['metrics'].items():
                        tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                    current_iter)
                        updated.append(tmp_updated)
                    # save best model if any metric is updated
                    if len(updated) > 0:
                        self.copy_model(self.net_g, self.net_g_best)
                        self.save_network(self.net_g, 'net_g_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'm_{metric}', value, current_iter)


    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_network(self.net_d, 'net_d', current_iter, epoch)
        self.save_training_state(epoch, current_iter)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
