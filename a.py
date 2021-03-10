import torch
from torchvision.transforms import functional
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from patch import GaussianBlurConv

image = Image.open('images/a.jpg')
image = functional.pil_to_tensor(image) / 255.
patch = Image.open('images/b.png')
patch = functional.pil_to_tensor(patch)
patch = functional.resize(patch, [220, 220])
x = 800
y = 650



# print(image)
# shadow = 132
# diff = 255 - shadow
# coe = 255.0 / diff
# rgb_diff = image - shadow
# rgb_diff = torch.clamp(rgb_diff, 0)
# img = rgb_diff * coe
# img = img[0]
# plt.imshow(np.array(functional.to_pil_image(image[0])))
# plt.show()
# plt.imshow(np.array(functional.to_pil_image(image2[0])))
# plt.show()
