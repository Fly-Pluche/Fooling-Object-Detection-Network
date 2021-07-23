import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image


class TensorToNumpy(nn.Module):
    """
    float tensor image to numpy uint8 image
    """

    def __init__(self):
        super(TensorToNumpy, self).__init__()

    def forward(self, image_tensor):
        image_rgb = torchvision.transforms.ToPILImage()(image_tensor)
        image_rgb = np.asarray(image_rgb, dtype='uint8')
        return image_rgb


def CMYK2RGB(img):
    """
    transform CMYK img to RGB img
    """

    img_rgb = torch.cuda.FloatTensor(3, img.size(1), img.size(2))
    img_rgb[0] = (1 - img[0]) * (1 - img[3])
    img_rgb[1] = (1 - img[1]) * (1 - img[3])
    img_rgb[2] = (1 - img[2]) * (1 - img[3])
    print(img_rgb)
    return img_rgb


if __name__ == '__main__':
    a = torch.full((3, 200, 200), 0.5)
    print(TensorToNumpy()(a))
