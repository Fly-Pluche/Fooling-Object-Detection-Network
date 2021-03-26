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
# train3 = 'datasets/train3.txt'
# train31 = 'datasets/train31.txt'
# with open(train3, 'r') as f:
#     train3 = f.readlines()
#
# with open(train31, 'r') as f:
#     train31 = f.readlines()
# train31 = [item.strip() for item in train31]
# num = 0
# print(len(train31))
# for i, item in enumerate(train3):
#     if item.strip() in train31:
#         train3.pop(i)
#         print(item.strip())
#         num += 1
#
# e = "".join(train3)
# with open('datasets/train_.txt', 'w') as f:
#     f.write(e)
# 
# import pyautogui as pg
# import time


# while True:
#     print(pg.position())


# pg.moveTo(3178, 469, 0.2)
# while True:
#     pg.moveTo(3350, 1250, 0.2)
#     pg.click()
#     time.sleep(10)
#     pg.moveTo(3178, 459, 0.2)
#     pg.click()
#     pg.moveTo(3070, 190, 0.2)
#     pg.click()
#     pg.moveTo(2751, 326, 0.2)
#     pg.click()
#     pg.dragTo(3409, 326, 0.2, button='left')

# import requests
# import json
# headers = {"Accept": "application/json, text/plain, */*",
#            "Cache-Control": "no-cache",
#            "Connection": "keep-alive",
#            "Content-Length": "23",
#            "Content-Type": "application/json;charset=UTF-8",
#            "Host": "kexieweb.guet.edu.cn",
#            "Origin": "http://kexieweb.guet.edu.cn",
#            "Pragma": "no-cache",
#            "Referer": "http://kexieweb.guet.edu.cn/",
#            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36", }
#
# data = {"userId": "1900300223"}
# data = json.dumps(data)
# a = requests.post(url='http://kexieweb.guet.edu.cn/api/user/signIn', headers=headers, data=data).text
# print(a)


for i in range(0,20):
    b = 6 ** i
    while (b * i % 41 != 1):
        i += 1

    print(i * 11 % 41)

# for i in range(1, 8):
#     print(29 ** i % 41)
