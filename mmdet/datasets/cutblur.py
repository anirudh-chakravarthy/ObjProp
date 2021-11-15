# Adapted from https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/augmentation/cutblur.py
import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class CutBlur(DataAugment):
    """CutBlur data augmentation, adapted from https://arxiv.org/abs/2004.00448.

    Randomly downsample a cuboid region in the volume to force the model
    to learn super-resolution when making predictions.

    Args:
        length_ratio (float): the ratio of the cuboid length compared with volume length.
        down_ratio_min (float): minimal downsample ratio to generate low-res region.
        down_ratio_max (float): maximal downsample ratio to generate low-res region.
        p (float): probability of applying the augmentation.
    """

    def __init__(self, 
                 length_ratio=0.25, 
                 down_ratio_min=2.0,
                 down_ratio_max=8.0,
                 p=0.5):
        super(CutBlur, self).__init__(p=p)
        self.length_ratio = length_ratio
        self.down_ratio_min = down_ratio_min
        self.down_ratio_max = down_ratio_max

    def set_params(self):
        # No change in sample size
        pass

    def cut_blur(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        yl, yh = self.random_region(images.shape[1], random_state)
        xl, xh = self.random_region(images.shape[2], random_state)

        temp = images[:, yl:yh, xl:xh].copy()

        down_ratio = random_state.uniform(self.down_ratio_min, self.down_ratio_max)
        out_shape = np.array(temp.shape) /  np.array([1, down_ratio, down_ratio])

        out_shape = out_shape.astype(int)
        downsampled = resize(temp, out_shape, order=1, mode='reflect', 
                             clip=True, preserve_range=True, anti_aliasing=True)
        upsampled = resize(downsampled, temp.shape, order=0, mode='reflect', 
                             clip=True, preserve_range=True, anti_aliasing=False)

        images[:, yl:yh, xl:xh] = upsampled
        return images, labels


    def random_region(self, vol_len, random_state):
        cuboid_len = int(self.length_ratio * vol_len)
        low = random_state.randint(0, vol_len-cuboid_len)
        high = low + cuboid_len
        return low, high

    def __call__(self, data, random_state=np.random):
        new_images, new_labels = self.cut_blur(data, random_state)
        return {'image': new_images, 'label': new_labels}
