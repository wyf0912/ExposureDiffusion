import numpy as np
import scipy.stats as stats
from os.path import join
from scipy.stats import tukeylambda

class RawPacker:
    def __init__(self, cfa='bayer'):
        self.cfa = cfa

    def pack_raw_bayer(self, cfa_img):
        # pack Bayer image to 4 channels
        img_shape = cfa_img.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((cfa_img[0:H:2, 0:W:2], # RGBG
                        cfa_img[0:H:2, 1:W:2],
                        cfa_img[1:H:2, 1:W:2],
                        cfa_img[1:H:2, 0:W:2]), axis=0).astype(np.float32)
        return out
    
    def pack_raw_xtrans(self, cfa_img):
        # pack X-Trans image to 9 channels
        img_shape = cfa_img.shape
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((9, H // 3, W // 3), dtype=np.float32)

        # 0 R
        out[0, 0::2, 0::2] = cfa_img[0:H:6, 0:W:6]
        out[0, 0::2, 1::2] = cfa_img[0:H:6, 4:W:6]
        out[0, 1::2, 0::2] = cfa_img[3:H:6, 1:W:6]
        out[0, 1::2, 1::2] = cfa_img[3:H:6, 3:W:6]

        # 1 G
        out[1, 0::2, 0::2] = cfa_img[0:H:6, 2:W:6]
        out[1, 0::2, 1::2] = cfa_img[0:H:6, 5:W:6]
        out[1, 1::2, 0::2] = cfa_img[3:H:6, 2:W:6]
        out[1, 1::2, 1::2] = cfa_img[3:H:6, 5:W:6]

        # 1 B
        out[2, 0::2, 0::2] = cfa_img[0:H:6, 1:W:6]
        out[2, 0::2, 1::2] = cfa_img[0:H:6, 3:W:6]
        out[2, 1::2, 0::2] = cfa_img[3:H:6, 0:W:6]
        out[2, 1::2, 1::2] = cfa_img[3:H:6, 4:W:6]

        # 4 R
        out[3, 0::2, 0::2] = cfa_img[1:H:6, 2:W:6]
        out[3, 0::2, 1::2] = cfa_img[2:H:6, 5:W:6]
        out[3, 1::2, 0::2] = cfa_img[5:H:6, 2:W:6]
        out[3, 1::2, 1::2] = cfa_img[4:H:6, 5:W:6]

        # 5 B
        out[4, 0::2, 0::2] = cfa_img[2:H:6, 2:W:6]
        out[4, 0::2, 1::2] = cfa_img[1:H:6, 5:W:6]
        out[4, 1::2, 0::2] = cfa_img[4:H:6, 2:W:6]
        out[4, 1::2, 1::2] = cfa_img[5:H:6, 5:W:6]

        out[5, :, :] = cfa_img[1:H:3, 0:W:3]
        out[6, :, :] = cfa_img[1:H:3, 1:W:3]
        out[7, :, :] = cfa_img[2:H:3, 0:W:3]
        out[8, :, :] = cfa_img[2:H:3, 1:W:3]
        return out

    def unpack_raw_bayer(self, img):        
        # unpack 4 channels to Bayer image
        img4c = img
        _, h, w = img.shape

        H = int(h * 2)
        W = int(w * 2)

        cfa_img = np.zeros((H, W), dtype=np.float32)

        cfa_img[0:H:2, 0:W:2] = img4c[0, :,:]
        cfa_img[0:H:2, 1:W:2] = img4c[1, :,:]
        cfa_img[1:H:2, 1:W:2] = img4c[2, :,:]
        cfa_img[1:H:2, 0:W:2] = img4c[3, :,:]
        
        return cfa_img

    def unpack_raw_xtrans(self, img):        
        img9c = img
        _, h, w = img.shape
        
        H = int(h * 3)
        W = int(w * 3)

        cfa_img = np.zeros((H, W), dtype=np.float32)

        # 0 R
        cfa_img[0:H:6, 0:W:6] = img9c[0, 0::2, 0::2]
        cfa_img[0:H:6, 4:W:6] = img9c[0, 0::2, 1::2]
        cfa_img[3:H:6, 1:W:6] = img9c[0, 1::2, 0::2]
        cfa_img[3:H:6, 3:W:6] = img9c[0, 1::2, 1::2]

        # 1 G
        cfa_img[0:H:6, 2:W:6] = img9c[1, 0::2, 0::2]
        cfa_img[0:H:6, 5:W:6] = img9c[1, 0::2, 1::2]
        cfa_img[3:H:6, 2:W:6] = img9c[1, 1::2, 0::2]
        cfa_img[3:H:6, 5:W:6] = img9c[1, 1::2, 1::2]

        # 1 B
        cfa_img[0:H:6, 1:W:6] = img9c[2, 0::2, 0::2]
        cfa_img[0:H:6, 3:W:6] = img9c[2, 0::2, 1::2]
        cfa_img[3:H:6, 0:W:6] = img9c[2, 1::2, 0::2]
        cfa_img[3:H:6, 4:W:6] = img9c[2, 1::2, 1::2]

        # 4 R
        cfa_img[1:H:6, 2:W:6] = img9c[3, 0::2, 0::2]
        cfa_img[2:H:6, 5:W:6] = img9c[3, 0::2, 1::2] 
        cfa_img[5:H:6, 2:W:6] = img9c[3, 1::2, 0::2] 
        cfa_img[4:H:6, 5:W:6] = img9c[3, 1::2, 1::2] 

        # 5 B
        cfa_img[2:H:6, 2:W:6] = img9c[4, 0::2, 0::2]
        cfa_img[1:H:6, 5:W:6] = img9c[4, 0::2, 1::2]
        cfa_img[4:H:6, 2:W:6] = img9c[4, 1::2, 0::2]
        cfa_img[5:H:6, 5:W:6] = img9c[4, 1::2, 1::2]

        cfa_img[1:H:3, 0:W:3] = img9c[5, :, :]
        cfa_img[1:H:3, 1:W:3] = img9c[6, :, :]
        cfa_img[2:H:3, 0:W:3] = img9c[7, :, :]
        cfa_img[2:H:3, 1:W:3] = img9c[8, :, :]
        
        return cfa_img    
    
    def pack_raw(self, cfa_img):
        if self.cfa == 'bayer':
            out = self.pack_raw_bayer(cfa_img)
        elif self.cfa == 'xtrans':
            out = self.pack_raw_xtrans(cfa_img)
        else:
            raise NotImplementedError
        return out

    def unpack_raw(self, img):
        if self.cfa == 'bayer':
            out = self.unpack_raw_bayer(img)
        elif self.cfa == 'xtrans':
            out = self.unpack_raw_xtrans(img)
        else:
            raise NotImplementedError        
        return out


class NoiseModelBase:  # base class
    def __call__(self, y, params=None, continuous=False):
        if params is None:
            K, g_scale, saturation_level, ratio, sigma_r, G_scale, G_shape = self._sample_params(continuous)
        else:
            K, g_scale, saturation_level, ratio, sigma_r, G_scale, G_shape = params

        y = y * saturation_level
        y = y / ratio
        if "u" in  self.model: # quantization noise
            y = y + (np.random.uniform(0, 1, y.shape) - 0.5)
            y = y.clip(0)
        if 'P' in self.model:
            z = np.random.poisson(y / K).astype(np.float32) * K
        elif 'p' in self.model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
        else:
            z = y

        if 'r' in self.model: # row noise
            z = self.raw_packer.unpack_raw(z)
            z = z + np.tile(np.random.randn(z.shape[0])[:,np.newaxis], (1, z.shape[1])).astype(np.float32) * np.maximum(sigma_r, 1e-10)
            z = self.raw_packer.pack_raw(z)
        
        if 'g' in self.model:
            z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10)  # Gaussian noise
        elif 'G' in self.model:
            lam = np.random.choice(G_shape)
            z = z + tukeylambda.rvs(lam, size=z.shape).astype("float32") * np.maximum(G_scale, 1e-10)
        else:
            z = z

        z = z * ratio
        z = z / saturation_level
        return z, {"K": K, "ratio": ratio, "g_scale": g_scale, "saturation_level": saturation_level}
    


# Only support baseline noise models: G / G+P / G+P* 
class NoiseModel(NoiseModelBase):
    def __init__(self, model='g', cameras=None, include=None, exclude=None, cfa='bayer'):
        super().__init__()
        assert cfa in ['bayer', 'xtrans']
        assert include is None or exclude is None
        self.cameras = cameras or ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']        

        if include is not None:
            self.cameras = [self.cameras[include]]
        if exclude is not None:
            exclude_camera = set([self.cameras[exclude]])
            self.cameras = list(set(self.cameras) - exclude_camera)

        self.param_dir = join('camera_params', 'release')

        print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] cameras: {}'.format(self.cameras))
        print('[i] using noise model {}'.format(model))
        
        self.camera_params = {}

        for camera in self.cameras:
            self.camera_params[camera] = np.load(join(self.param_dir, camera+'_params.npy'), allow_pickle=True).item()

        self.model = model
        self.raw_packer = RawPacker(cfa)
        
    def ISO_to_K(self, ISO):
        camera_params = self.camera_params[self.cameras[0]]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        k = (ISO - 100)/(6400-100) *(Kmax-Kmin)+Kmin
        return k
    
    def _sample_params(self, continuous):
        camera = np.random.choice(self.cameras)
        # print(camera)

        saturation_level = 16383 - 800
        profiles = ['Profile-1']

        camera_params = self.camera_params[camera]
        
        G_shape = camera_params["G_shape"]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        # log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        log_K = np.random.uniform(low=np.log(1e-1), high=np.log(30))
        
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
             camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']

        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)
        if continuous:
            ratio = np.random.uniform(low=20, high=300)
        else:
            ratio = np.random.uniform(low=100, high=300)
            
        log_r = np.random.standard_normal() * camera_params['R_scale']['sigma'] * 1 +\
             camera_params['R_scale']['slope'] * log_K + camera_params['R_scale']['bias']
        log_G_scale = np.random.standard_normal() * camera_params['G_scale']['sigma'] * 1 +\
             camera_params['G_scale']['slope'] * log_K + camera_params['G_scale']['bias']
        sigma_r = np.exp(log_r)
        G_scale = np.exp(log_G_scale)
        return (K, g_scale, saturation_level, ratio, sigma_r, G_scale, G_shape)
