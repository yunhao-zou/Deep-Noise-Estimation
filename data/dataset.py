from sys import meta_path
import torch
import os
import glob
import rawpy
import numpy as np
import random
from os.path import join
import data.torchdata as torchdata
import util.process as process
# from util.image_pipeline import cfa_pattern
# from data.lmdb_dataset import LMDBDataset
# from util.util import loadmat
from scipy.io import loadmat
import h5py
import exifread
import pickle
import tifffile as tiff
import PIL.Image as Image
from torch.multiprocessing import Manager
from torchvision import transforms
from scipy.io import loadmat
import re
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
BaseDataset = torchdata.Dataset


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def read_metadata(metadata):
    meta = metadata['metadata'][0, 0]
    cam = get_cam(meta)
    bayer_pattern = get_bayer_pattern(meta)
    # We found that the correct Bayer pattern is GBRG in S6
    if cam == 'S6':
        bayer_pattern = [1, 2, 0, 1]
    bayer_2by2 = (np.asarray(bayer_pattern) + 1).reshape((2, 2)).tolist()
    wb = get_wb(meta)
    cst1, cst2 = get_csts(meta) # use cst2 for rendering
    iso = get_iso(meta)
    
    return meta, bayer_2by2, wb, cst2, iso, cam

def get_iso(metadata):
    try:
        iso = metadata['ISOSpeedRatings'][0][0]
    except:
        try:
            iso = metadata['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        except:
            raise Exception('ISO not found.')
    return iso


def get_cam(metadata):
    model = metadata['Make'][0]
    cam_dict = {'Apple': 'IP', 'Google': 'GP', 'samsung': 'S6', 'motorola': 'N6', 'LGE': 'G4'}
    return cam_dict[model]


def get_bayer_pattern(metadata):
    bayer_id = 33422
    bayer_tag_idx = 1
    try:
        unknown_tags = metadata['UnknownTags']
        if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
            bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
        else:
            raise Exception
    except:
        try:
            unknown_tags = metadata['SubIFDs'][0, 0]['UnknownTags'][0, 0]
            if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
                bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
            else:
                raise Exception
        except:
            try:
                unknown_tags = metadata['SubIFDs'][0, 1]['UnknownTags']
                if unknown_tags[1]['ID'][0][0][0] == bayer_id:
                    bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
                else:
                    raise Exception
            except:
                print('Bayer pattern not found. Assuming RGGB.')
                bayer_pattern = [1, 2, 2, 3]
    return bayer_pattern


def get_wb(metadata):
    return metadata['AsShotNeutral']


def get_csts(metadata):
    return metadata['ColorMatrix1'].reshape((3, 3)), metadata['ColorMatrix2'].reshape((3, 3))

def transform_pattern(pattern):
    pattern[pattern==3] = 0
    pattern[pattern==1] = 4
    pattern[pattern==2] = [1, 3]
    pattern[pattern==4] = 2
    return pattern



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

class SiddEvalDataset(BaseDataset):
    def __init__(self, datadir=None, camera=None, size=None, amp_ratio=1, repeat=1, srgb=False):
        super(SiddEvalDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        # self.fns = fns or [fn for fn in os.listdir(datadir) if os.path.isfile(join(datadir, fn))]
        self.fns = sorted(glob.glob(join(datadir, '*{}*'.format(camera), '*NOISY_RAW*.MAT')))
        self.amp_ratio = amp_ratio
        self.srgb = srgb
        self.repeat = repeat
        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, index):
        # np.random.seed()
        index = index % len(self.fns)
        input_fn = self.fns[index]
        input_path = input_fn
        meta_path = input_path.replace('NOISY_RAW', 'METADATA_RAW')
        f = h5py.File(input_path, 'r')
        # meta_info = h5py.File('/data/noise_flow/data/SIDD_Medium_Raw/Data/0001_001_S6_00100_00060_3200_L/0001_METADATA_RAW_010.MAT', 'r')
        meta_info = loadmat(meta_path)
        meta, bayer_2by2, wb, cst2, iso, cam = read_metadata(meta_info)
        bayer_raw = f['x'][:]
        # pattern = np.array([[1,0], [2,3]])
        pattern = transform_pattern(np.array(bayer_2by2))
        packed_raw = pack_raw_bayer(bayer_raw, pattern, 0, 1)
        crop = 512
        c, h, w = packed_raw.shape
        x = np.random.randint(0, h-crop)
        y = np.random.randint(0, w-crop)
        # x = (h-crop) // 2
        # y = (w-crop) // 2
        input = packed_raw[:, x:x+crop, y:y+crop]
        input = np.clip(packed_raw, 0, 1)
        input = np.ascontiguousarray(input)
        
        data = {'input': input, 'rawpath': input_path, 'metapath': meta_path, 'iso': iso.astype(np.int64)}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size) * self.repeat
        else:
            return len(self.fns) * self.repeat