import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import cv2
from colour_demosaicing import *

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,0,...], 
        torch.mean(bayer_images[:, [1,3], ...], dim=1), 
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def demosaic(in_vid, converter=cv2.COLOR_BayerBG2RGB):
    bayer_input = np.zeros([in_vid.shape[0], in_vid.shape[2] * 2, in_vid.shape[3] * 2], dtype=np.float32)
    bayer_input[:, ::2, ::2] = in_vid[:, 2, :, :]
    bayer_input[:, ::2, 1::2] = in_vid[:, 1, :, :]
    bayer_input[:, 1::2, ::2] = in_vid[:, 3, :, :]
    bayer_input[:, 1::2, 1::2] = in_vid[:, 0, :, :]
    # bayer_input = (bayer_input * 65535).astype('uint16')
    rgb_input = np.zeros([bayer_input.shape[0], bayer_input.shape[1], bayer_input.shape[2], 3], dtype=np.float32)
    for j in range(bayer_input.shape[0]):
        # rgb_input[j] = cv2.cvtColor(bayer_input[j], converter)
        rgb_input[j] = demosaicing_CFA_Bayer_DDFAPD(bayer_input[j], 'BGGR')
    rgb_input = rgb_input.transpose((0, 3, 1, 2))
    rgb_input = torch.from_numpy(rgb_input)
    return rgb_input


def process(bayer_images, wbs, cam2rgbs, gamma=2.2, use_demosaic=False):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, wbs)
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    if use_demosaic:
        images = demosaic(bayer_images)
    else:
        images = binning(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)
    images = torch.clamp(images, min=0.0, max=1.0)
    return images


def raw2rgb(packed_raw, raw): 
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    if isinstance(packed_raw, np.ndarray):
        packed_raw = torch.from_numpy(packed_raw).float()

    wb = torch.from_numpy(wb).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2)[0, ...].numpy()
    
    return out


def raw2rgb_v2(packed_raw, wb, ccm):
    packed_raw = torch.from_numpy(packed_raw).float()
    wb = torch.from_numpy(wb).float()
    cam2rgb = torch.from_numpy(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2)[0, ...].numpy()
    return out


def raw2rgb_postprocess(packed_raw, raw):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4 and packed_raw.shape[0] == 1
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    wb = torch.from_numpy(wb[None]).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb[None]).float().to(packed_raw.device)
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2)
    return out


def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.rgb_camera_matrix[:3, :3].astype(np.float32)
    return wb, ccm
