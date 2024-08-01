import os
from struct import unpack
import cv2
import math
import glob
import yaml
import random
import shutil
import argparse
import numpy as np
import scipy.io as sio
import h5py
import torch

import noise_model
from util.utils import toTensor, read_metadata, process_sidd_image, cal_kld


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str)
    parser.add_argument('--camera', default='S6', type=str)
    return parser.parse_args()

def pack_raw_bayer(bayer_raw, raw_pattern, bl, wl):
    #pack Bayer image to 4 channels
    im = bayer_raw
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    white_point = wl
    black_level = bl
    
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out

def unpack_bayer(packed_raw, raw_pattern):
    img4c = packed_raw
    _, h, w = packed_raw.shape
    H, W = 2 * h, 2 * w
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    out = np.zeros([2*h, 2*w], dtype=np.float32)

    out[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    out[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    out[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    out[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]

    return out

def addText(img, kl=None):
    # kl=None
    a, b, _ = img.shape
    scale = 512 // a
    noisyne_patch_srgb = cv2.resize(img, (a * scale, b * scale), interpolation=cv2.INTER_NEAREST)
    if kl:
        x, y, w, h = 0, 0, 256, 65
        sub_img = noisyne_patch_srgb[y:y+h, x:x+w]
        white_rect = np.ones(sub_img.shape, dtype=np.float64) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.7, 1.0)
        # Putting the image back to its position
        noisyne_patch_srgb[y:y+h, x:x+w] = res
        cv2.putText(noisyne_patch_srgb, 'KL={:.3f}'.format(kl), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,39) ,3)
    return noisyne_patch_srgb

def main():
    args = parse_args()
    with open(args.config, 'rt') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    folder_png = cfg['noise_model']['sample_dir_png']
    if os.path.exists(folder_png):
        shutil.rmtree(folder_png)
    os.makedirs(folder_png, exist_ok=True)

    # for separate ISO sampling
    add_noise = noise_model.NoiseModel(cameras=[args.camera], min_ratio=1, max_ratio=1, sigma_rate=0, linear=False, param_path='./sidd_params/{}.npy'.format(args.camera))

    # for joint ditribution (continuous ISO)
    # add_noise = noise_model.NoiseModel(cameras=[args.camera], include=None, exclude=None, min_ratio=1, max_ratio=1, sigma_rate=0, linear=True, param_path='./sidd_params/{}.npy'.format(args.camera)) 

    for data_idx, data_file in enumerate(glob.glob(os.path.join(cfg['data_dir'],'*[!_meta].mat'))):
        data = sio.loadmat(data_file)
        clean, noisy = data['clean'], data['noisy']
        metadata = sio.loadmat('%s_meta.mat' % data_file[:-4])
        meta, bayer_2by2, wb, cst2, iso, cam = read_metadata(metadata)
        if str(cam) != str(args.camera):
            continue
        # For each sample
        ps = cfg['noise_model']['patch_size']
        for sample_idx in range(cfg['noise_model']['sample_amount']):
            # Log
            print('%s [%04d] [ISO %04d]' %(os.path.basename(data_file), sample_idx, iso))
            # Crop patch
            x = random.randrange(0, clean.shape[0]-ps[0], 2) # 2 for Bayer pattern
            y = random.randrange(0, clean.shape[1]-ps[1], 2) # 2 for Bayer pattern
            clean_patch = clean[x:x+ps[0], y:y+ps[1]]
            noisy_patch = noisy[x:x+ps[0], y:y+ps[1]]
            noise_patch = noisy_patch - clean_patch
            pattern = np.array(bayer_2by2)
            pattern[pattern==2] = [2, 4]
            pattern -= 1
            clean_bayer = pack_raw_bayer(clean_patch, pattern, 0, 1)
            noisyne_patch, _ = add_noise(clean_bayer, iso)
            noisyne_patch = np.clip(noisyne_patch, 0.0, 1.0)
            noisyne_patch = unpack_bayer(noisyne_patch, pattern)
            ne_patch = noisyne_patch - clean_patch
            ## Log (KL divergence)
            kld = cal_kld(noise_patch, ne_patch)
            print("[KLD] Ours: %.4f"%(kld))
            clean_patch_srgb = process_sidd_image(clean_patch, bayer_2by2, wb, cst2)
            clean_patch_srgb = addText(clean_patch_srgb[1:-1, 1:-1, :])
            noisy_patch_srgb = process_sidd_image(noisy_patch, bayer_2by2, wb, cst2)
            noisy_patch_srgb = addText(noisy_patch_srgb[1:-1, 1:-1, :])
            ne_patch_srgb = process_sidd_image(ne_patch, bayer_2by2, wb, cst2)
            ne_patch_srgb = addText(ne_patch_srgb[1:-1, 1:-1, :])
            noisyne_patch_srgb = process_sidd_image(noisyne_patch, bayer_2by2, wb, cst2)
            noisyne_patch_srgb = addText(noisyne_patch_srgb[1:-1, 1:-1, :])
            cv2.imwrite(os.path.join(folder_png, '%s_%03d_%03d_%04d_clean.png' % (args.camera, data_idx, sample_idx, iso)), clean_patch_srgb)
            cv2.imwrite(os.path.join(folder_png, '%s_%03d_%03d_%04d_real.png' % (args.camera, data_idx, sample_idx, iso)), noisy_patch_srgb)
            cv2.imwrite(os.path.join(folder_png, '%s_%03d_%03d_%04d_synthesis.png' % (args.camera, data_idx, sample_idx, iso)), noisyne_patch_srgb)

if __name__ == "__main__":
    main()