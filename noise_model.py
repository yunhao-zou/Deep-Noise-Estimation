import numpy as np
from numpy.core.fromnumeric import size
import scipy.stats as stats
from os.path import join

class NoiseModel:
    def __init__(self, cameras=None, min_ratio=10, max_ratio=10, sigma_rate=1, linear=True, param_path=None):
        super().__init__()
        self.cameras = cameras
        self.param_dir = join('sidd_params')
        self.value = None

        self.camera_params = {}
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.sigma_rate = sigma_rate
        self.linear = linear
        self.param_path = param_path
        for camera in self.cameras:
            self.camera_params[camera] = np.load(join(self.param_dir, camera+'_params.npy'), allow_pickle=True).item()

    def _sample_params(self, iso):
        camera = np.random.choice(self.cameras)
        Q_step = 1
        saturation_level = 1023 - 256
        profiles = ['Profile-1']

        camera_params = self.camera_params[camera]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        if self.linear:
            # sample from joint distribution
            if iso is None:
                log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
                K = np.exp(log_K)
                iso = K / camera_params['iso2K']
            else:
                K = iso * camera_params['iso2K']
                profile = np.random.choice(profiles)
                camera_params = camera_params[profile]
                log_K = np.log(K)
                log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * self.sigma_rate +\
                    camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']
                log_R_scale = np.random.standard_normal() * camera_params['R_scale']['sigma'] * self.sigma_rate +\
                    camera_params['R_scale']['slope'] * log_K + camera_params['R_scale']['bias']
                params = np.load(self.param_path)
                ind = np.random.randint(0, params.shape[0])
                color_bias = params[ind, 3:7]
        else:
            # sample from separate noise parameters
            params = np.load(self.param_path)
            log_K = np.median(params[np.argwhere((params[:, -1]).astype(int)==iso), 0])
            K = np.exp(log_K)
            log_g_scale = np.median(params[np.argwhere(params[:, -1].astype(int)==iso), 1])
            log_R_scale = np.median(params[np.argwhere(params[:, -1].astype(int)==iso), 2])
            ind = np.random.randint(0, params.shape[0])
            color_bias = params[ind, 3:7]
        g_scale = np.exp(log_g_scale)
        R_scale = np.exp(log_R_scale)

        ratio = np.random.uniform(low=self.min_ratio, high=self.max_ratio)
        return np.array([K, color_bias, g_scale, R_scale, Q_step, saturation_level, ratio, iso], dtype=object) 

    def __call__(self, y, iso=None, value=None, params=None):
        if params is None:
            params = self._sample_params(iso)
        if value is not None:
            self.value = value
        K, color_bias, g_scale, R_scale, Q_step, saturation_level, ratio, iso = params
        y = y * saturation_level
        y = y / ratio
        z = np.random.poisson(y / K).astype(np.float32) * K  # shot noise
        z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10) # read noise
        z = self.add_color_bias(z, color_bias=color_bias)  # color bias
        z = self.add_banding_noise(z, scale=R_scale)  # row noise
        z = z + np.random.uniform(low=-0.5*Q_step, high=0.5*Q_step) # quantization noise

        z = z * ratio
        z = z / saturation_level
        return z, params

    def add_color_bias(self, img, color_bias):
        channel = img.shape[0]
        img = img + color_bias.reshape((channel,1,1))
        return img

    def add_banding_noise(self, img, scale):
        channel = img.shape[0]
        img = img + np.random.randn(channel, img.shape[1], 1).astype(np.float32) * scale
        return img

