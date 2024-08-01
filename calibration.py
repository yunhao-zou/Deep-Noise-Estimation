import pickle
import numpy as np
from numpy.core.defchararray import index
import rawpy
import exifread
import os
from os.path import join, splitext
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import cv2
import scipy.stats as stats
from scipy.io import savemat, loadmat
from scipy.optimize import least_squares, minimize, curve_fit
import seaborn as sns
sns.set(color_codes=True)
from collections import namedtuple
import glob


def joint_params_dist(params, camera):
    iso_min = np.min(params[:, -1])
    iso_max = np.max(params[:, -1])
    K_min = np.exp(np.min(params[:, 0]))
    K_max = np.exp(np.max(params[:, 0]))
    iso2K = K_max / iso_max
    camera_params = {}
    camera_params['Kmin'] = K_min
    camera_params['Kmax'] = K_max
    camera_params['iso2K'] = iso2K
    log_K = params[:, 0]  # log_K
    labels = ['iso', 'g_scale', 'R_scale', 'color_bias']
    camera_params['Profile-1'] = {}

    for i in [1, 2]:
        # calibrate g_scale and R_scale
        y = params[:, i]  
        x = log_K

        slope, intercept, r, prob, sterrest = stats.linregress(x, y)
        y_est = x * slope + intercept
        rss = np.sum((y - y_est)**2)
        err_std = np.sqrt(rss / (len(y) - 2))
        print(slope, intercept, err_std)

        camera_params['Profile-1'][labels[i]] = {}
        camera_params['Profile-1'][labels[i]]['slope'] = slope
        camera_params['Profile-1'][labels[i]]['bias'] = intercept
        camera_params['Profile-1'][labels[i]]['sigma'] = err_std

        ax = plt.subplot(111)
        sns.regplot(x, y, 'ci', ax=ax)
        plt.plot(x, y, 'bo', x, y_est, 'r-')
        plt.xlabel('$\log(K)$', fontsize=18)
        if i == 1:
            plt.ylabel('$\log(\sigma_{read})$', fontsize=18)
        elif i == 2:
            plt.ylabel('$\log(\sigma_{row})$', fontsize=18)

        plt.savefig('./plot_calib/{}_{}.png'.format(camera, i), bbox_inches='tight')
        print('saved in ./plot_calib/{}_{}.png'.format(camera, i))
        plt.clf()

    return camera_params


if __name__ == '__main__':
    param_path = '/data/NoiseEstimate/sidd_params/S6.npy'
    params = np.load(param_path)
    camera_params = joint_params_dist(params, camera='S6')
    params_path = join('sidd_params', 'S6_params.npy')
    if not os.path.exists(params_path):
        print('save {}..'.format(params_path))
        np.save(params_path, camera_params)