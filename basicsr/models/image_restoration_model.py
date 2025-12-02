# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import copy
import time
import torch
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import random

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class Mixing_Augment:
    """
    Mixup augmentation for image restoration.
    Args:
        mixup_beta (float): The beta parameter of Beta distribution.
        use_identity (bool): Whether to allow not using mixup.
        device (torch.device): The device of the model.
    """
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # mixup augmentation
        if opt['is_train']:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        else:
            self.mixing_flag = False
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        print_network = opt.get('print_network', False)
        if print_network:
            self.print_network(self.net_g)

        if self.is_train:
            self.init_training_settings()
            self.net_g_best = copy.deepcopy(self.net_g)

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # whether to freeze (for llm arch)
        if self.opt['train']['freeze_except_adapter']:
            for name, param in self.net_g.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad = False

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # if train_opt.get('perceptual_opt'):
        #     percep_type = train_opt['perceptual_opt'].pop('type')
        #     cri_perceptual_cls = getattr(loss_module, percep_type)
        #     self.cri_perceptual = cri_perceptual_cls(
        #         **train_opt['perceptual_opt']).to(self.device)
        # else:
        #     self.cri_perceptual = None

        # fft_loss
        if train_opt.get('fft_opt'):
            pixel_type = train_opt['fft_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_fft = cri_pix_cls(**train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('ssim_opt'):
            pixel_type = train_opt['ssim_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_ssim = cri_pix_cls(**train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

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
            # if v.requires_grad:
            #     optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)


    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        # fft loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_freq'] = l_fft

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim


        # l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        # l_total.backward()
        l_total.mean().backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
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
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                if not self.opt['is_train']:
                    allocated_memory_before = torch.cuda.memory_allocated()
                    start_time = time.time()

                inference_uhd = self.opt.get('inference_uhd', False)
                if inference_uhd:
                    if hasattr(self.net_g, 'module'):
                        # for DDP
                        pred = self.net_g.module.test_tile(self.lq[i:j])
                    else:
                        pred = self.net_g.test_tile(self.lq[i:j])
                else:
                    pred = self.net_g(self.lq[i:j])

                if not self.opt['is_train']:
                    end_time = time.time()
                    allocated_memory_after = torch.cuda.memory_allocated()
                    self.inference_time += end_time - start_time
                    self.inference_memory = allocated_memory_after - allocated_memory_before

                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self._initialize_best_metric_results(dataset_name)

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        if not self.opt['is_train']:
            self.inference_time = 0.

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)

            # padding to multiple of 16 (specially for LLformer)
            if self.opt['val'].get('padding', False):
                self.padding()

            self.test()

            if self.opt['val'].get('padding', False):
                self.padding_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        # save_gt_img_path = osp.join(self.opt['path']['visualization'],
                        #                          img_name,
                        #                          f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        # save_gt_img_path = osp.join(
                        #     self.opt['path']['visualization'], dataset_name,
                        #     f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    # imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        opt_.pop('save_best_checkpoint')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        opt_.pop('save_best_checkpoint')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)

            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)
            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            # update each metric separately
            if self.opt['is_train']:
                for name, opt_ in self.opt['val']['metrics'].items():
                    updated = self._update_best_metric_result(dataset_name, name, metrics_dict[name], current_iter)

                    # save best model if target metric is updated
                    if opt_['save_best_checkpoint'] and updated:
                        self.copy_model(self.net_g, self.net_g_best)
                        self.save_network(self.net_g, 'net_g_best', '')

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \n'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        if not self.opt['is_train']:
            logger.info(f'Inference time: {self.inference_time:.2f}s')
            logger.info(f'Inference memory: {self.inference_memory / 1024 / 1024:.2f}MB')

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)