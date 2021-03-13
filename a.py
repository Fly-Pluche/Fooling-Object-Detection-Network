import torch
from torchvision.transforms import functional
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# from patch import GaussianBlurConv

# image = Image.open('images/a.jpg')
# image = functional.pil_to_tensor(image) / 255.
# patch = Image.open('images/b.png')
# patch = functional.pil_to_tensor(patch)
# patch = functional.resize(patch, [220, 220])
# x = 800
# y = 650


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


# x = list(range(1, 20, 2))
# y = [format(float(item ** (1 / 2)), '.2f') for item in x]
# print(y)
# y = []
# for item in x:
#     y.append(format(float(item ** (1 / 2)), '.2f'))
train3 = 'datasets/train3.txt'
train31 = 'datasets/train31.txt'
with open(train3, 'r') as f:
    train3 = f.readlines()

with open(train31, 'r') as f:
    train31 = f.readlines()
train31 = [item.strip() for item in train31]
num = 0
print(len(train31))
for i, item in enumerate(train3):
    if item.strip() in train31:
        train3.pop(i)
        print(item.strip())
        num += 1

e = "".join(train3)
with open('datasets/train_.txt', 'w') as f:
    f.write(e)
