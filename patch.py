from __future__ import absolute_import
import cv2
import torch
import math
import numpy as np
from torch import nn
from patch_config import patch_configs
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torchvision import transforms
import matplotlib.pyplot as plt


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class PatchTransformer(nn.Module):
    """
    change patch's contrast and brightness
    rotate the patch
    return a batch patch mask which have the same size with the training image.
    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8  # min contrast
        self.max_contrast = 1.2  # max contrast
        self.min_brightness = -0.1  # min brightness
        self.max_brightness = 0.1  # max brightness
        self.min_angle = -20 / 180 * math.pi  # min angle
        self.max_angle = 20 / 180 * math.pi  # max angle
        self.noise_factor = 0.10  # a limit to noise
        self.configs = patch_configs['base']()
        self.median_pooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, boxes_batch, lab_batch, people_id=0):
        # make people id be 1 other is 0
        for i in range(lab_batch.size()[0]):
            for j in range(lab_batch.size()[1]):
                if lab_batch[i, j] == people_id:
                    lab_batch[i, j] = 1
                else:
                    lab_batch[i, j] = 0
        # make a batch of adversarial patch
        adv_patch = self.median_pooler(adv_patch.unsqueeze(0))
        # determine size of padding
        img_size = np.array(self.configs.img_size)
        # adv_patch is a square
        pad = list((img_size - adv_patch.size(-1)) / 2)
        # a image needs boxes number patch
        adv_batch = adv_patch.unsqueeze(0)
        # a batch adv_batch: (batch size, boxes number, patch.size(0),patch.size(1),patch.size(2))
        adv_batch = adv_batch.expand(boxes_batch.size(0), boxes_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((boxes_batch.size(0), boxes_batch.size(1)))

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # apply contrast, brightness and clamp
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.999999)

        # pad patch and mask to image dimensions
        my_pad = nn.ConstantPad2d((int(pad[0] + 0.5), int(pad[0]), int(pad[1] + 0.5), int(pad[1])), 0)
        adv_batch = my_pad(adv_batch)

        # rotation and rescaling transforms
        angle_size = (lab_batch.size(0) * lab_batch.size(1))
        angle = torch.cuda.FloatTensor(angle_size).uniform_(self.min_angle, self.max_angle)

        # Resizes and rotates the patch
        current_patch_size = adv_patch.size(-1)
        boxes_batch_scaled = torch.cuda.FloatTensor(boxes_batch.size()).fill_(0)
        img_size = list(img_size)
        # box [x,y,w,h]
        boxes_batch_scaled[:, :, 0] = boxes_batch[:, :, 0] * img_size[0]
        boxes_batch_scaled[:, :, 1] = boxes_batch[:, :, 1] * img_size[1]
        boxes_batch_scaled[:, :, 2] = boxes_batch[:, :, 2] * img_size[0]
        boxes_batch_scaled[:, :, 3] = boxes_batch[:, :, 3] * img_size[1]
        target_size = torch.sqrt_(
            ((boxes_batch_scaled[:, :, 2].mul(0.2)) ** 2) + ((boxes_batch_scaled[:, :, 3].mul(0.2)) ** 2)
        )
        target_x = boxes_batch[:, :, 0].view(np.prod(batch_size))
        target_y = boxes_batch[:, :, 1].view(np.prod(batch_size))
        target_off_x = boxes_batch[:, :, 2].view(np.prod(batch_size))
        target_off_y = boxes_batch[:, :, 3].view(np.prod(batch_size))

        # random change the patches' position
        off_x = target_off_x * (torch.cuda.FloatTensor(target_off_x.size()).uniform_(-0.3, 0.3))
        target_x = target_x + off_x
        off_y = target_off_y * (torch.cuda.FloatTensor(target_off_y.size()).uniform_(-0.1, 0.1))
        target_y = target_y + off_y
        target_y = target_y - 0.01
        scale = target_size / current_patch_size
        scale = scale.view(angle_size)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # theta = roataion, rescale matrix
        theta = torch.cuda.FloatTensor(angle_size, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0, 0.999999)
        return adv_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        # adv_batch [4, 15, 3, 416, 416]
        for adv in advs:
            img_batch = torch.where((adv > 0.000001), adv, img_batch)
        return img_batch
