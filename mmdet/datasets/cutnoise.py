# Adapted from https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/cutnoise.py
import numpy as np
from .augmentor import DataAugment

class CutNoise(DataAugment):
    """3D CutNoise data augmentation.

    Randomly add noise to a cuboid region in the volume to force the model
    to learn denoising when making predictions.

    Args:
        length_ratio (float): the ratio of the cuboid length compared with volume length.
        mode (string): the distribution of the noise pattern. Default: ``'uniform'``.
        scale (float): scale of the random noise. Default: 0.2.
        p (float): probability of applying the augmentation.
    """

    def __init__(self, 
                 length_ratio=0.25, 
                 mode='uniform',
                 scale=0.2,
                 p=0.5,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 **kwargs):
        super(CutNoise, self).__init__(p=p)
        self.length_ratio = length_ratio
        self.mode = mode
        self.scale = scale
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)


    def set_params(self):
        # No change in sample size
        pass

    def cut_noise(self, data, random_state):
        images = data['image'].copy() # C x H x W
        labels = data['label'].copy()
        
        yl, yh = self.random_region(images.shape[1], random_state)
        xl, xh = self.random_region(images.shape[2], random_state)
        
        temp = images[:, yl:yh, xl:xh].copy()
        img_max = (255. - self.mean) / self.std
        img_min = (0. - self.mean) / self.std
        for i in range(images.shape[0]):
            scale = self.scale * (img_max[i] - img_min[i])
            noise = random_state.uniform(-scale, scale, temp[i].shape)
            temp[i] = temp[i] + noise
            temp[i] = np.clip(temp[i], img_min[i], img_max[i])

        images[:, yl:yh, xl:xh] = temp
        return images, labels

    def random_region(self, vol_len, random_state):
        cuboid_len = int(self.length_ratio * vol_len)
        low = random_state.randint(0, vol_len-cuboid_len)
        high = low + cuboid_len
        return low, high

    def __call__(self, data, random_state=np.random):
        new_images, new_labels = self.cut_noise(data, random_state)
        return {'image': new_images, 'label': new_labels}