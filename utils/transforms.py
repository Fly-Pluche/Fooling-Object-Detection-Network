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





if __name__ == '__main__':
    a = torch.full((3, 200, 200), 0.5)
    print(TensorToNumpy()(a))
