import os
import torch
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import pyiqa
import warnings
warnings.filterwarnings("ignore")

from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray

def normalize_img(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    if img1.max() > 1.0 or img2.max() > 1.0:
        img1 /= 255.0
        img2 /= 255.0

    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    return img1, img2

def calculate_mae(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    total_mae = np.mean(np.abs(img1 - img2))
    return total_mae

def metrics(im_dir, label_dir, use_GT_mean, use_y_channel, device):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_mae = 0
    n = 0

    psnr_model = pyiqa.create_metric('psnr', test_y_channel=use_y_channel, device=device)
    ssim_model = pyiqa.create_metric('ssim', test_y_channel=use_y_channel, device=device)
    lpips_model = pyiqa.create_metric('lpips', device=device)

    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        im1 = Image.open(item).convert('RGB')

        filename = os.path.splitext(os.path.basename(item))[0]
        possible_extensions = ['.png', '.jpg', '.jpeg']
        for ext in possible_extensions:
            try:
                im2_path = os.path.join(label_dir, filename + ext)
                im2 = Image.open(im2_path).convert('RGB')
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find a compatible file for {filename} in {label_dir} with extensions {possible_extensions}")

        (h, w) = im2.size
        im1 = im1.resize((h, w))
        im1 = np.array(im1)
        im2 = np.array(im2)

        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target/mean_restored), 0, 255)

        img1, img2 = normalize_img(im1, im2)
        if use_y_channel:
            score_psnr = psnr_gray(im1, im2)
            score_ssim = ssim_gray(im1, im2)
        else:
            score_psnr = psnr_model(img1, img2).item()
            score_ssim = ssim_model(img1, img2).item()

        score_lpips = lpips_model(img1, img2).item()
        score_mae = calculate_mae(im1, im2)

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips
        avg_mae += score_mae
        torch.cuda.empty_cache()


    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    avg_mae = avg_mae / n
    return avg_psnr, avg_ssim, avg_lpips, avg_mae


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=bool, default=True, help='save metrics')
    parser.add_argument("--y", action='store_true', help='use y channel')
    parser.add_argument("--gt", action='store_true', help='use gt mean')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    test_result = 'results/LLFormerv2_LOLv1/visualization/lol-test/*.png'
    gt = '/mnt/data/shenglong/data/LOL/eval15/high/'

    use_GT_mean = args.gt
    use_y_channel = args.y
    save_output = args.save

    avg_psnr, avg_ssim, avg_lpips, avg_mae = metrics(test_result, gt, use_GT_mean, use_y_channel, device)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
    print("===> Avg.MAE: {:.4f} ".format(avg_mae))

    # output metrics to json file
    im_list = test_result.split('/')
    if save_output:
        if use_GT_mean:
            file = "metric_gt_mean.json"
        elif use_y_channel:
            file = "metric_y_channel.json"
        else:
            file = "metric.json"
        output_file = os.path.join(im_list[0], im_list[1], file)
        metrics_dict = {
            "Avg.PSNR": avg_psnr,
            "Avg.SSIM": avg_ssim,
            "Avg.LPIPS": avg_lpips,
            "Avg.MAE": avg_mae
        }

        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        print(f"Metrics saved to: {output_file}")
