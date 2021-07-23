import torch
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
import numpy as np
import glob
import os
import json
from tqdm import tqdm
import cv2
from torchvision.transforms import functional


class DataAugmentation(object):
    def __init__(self):
        super(DataAugmentation, self).__init__()

    def adjust_anno_position(self, anno, x_offset, y_offset):
        for item in anno:
            if 'item' in item:
                landmarks = torch.tensor(anno[item]['landmarks'])
                landmarks = landmarks.view((-1, 3))
                landmarks[:, 0] = landmarks[:, 0] + x_offset
                landmarks[:, 1] = landmarks[:, 1] + y_offset
                landmarks = landmarks.view((landmarks.size(0) * landmarks.size(1)))
                landmarks = landmarks.numpy().tolist()
                anno[item]['landmarks'] = landmarks
        people_boxes = anno['people_boxes']
        people_boxes = torch.tensor(people_boxes)
        people_boxes[:, 0] = people_boxes[:, 0] + x_offset
        people_boxes[:, 2] = people_boxes[:, 2] + x_offset
        people_boxes[:, 1] = people_boxes[:, 1] + y_offset
        people_boxes[:, 3] = people_boxes[:, 3] + y_offset
        anno['people_boxes'] = people_boxes
        return anno

    def resize_(self, img, boxes, size):
        # -----------------------------------------------------------
        # 类型为 img=Image.open(path)，boxes:Tensor，size:int
        # 功能为：将图像短边缩放到指定值size,保持原有比例不变，并且相应调整boxes
        # -----------------------------------------------------------
        w, h = img.size
        min_size = min(w, h)
        sw = sh = size / min_size
        ow = int(sw * w + 0.5)
        oh = int(sh * h + 0.5)
        img = img.resize((ow, oh), Image.BILINEAR)
        return img.resize((ow, oh), Image.BILINEAR), boxes * torch.Tensor([sw, sh, sw, sh])

    def random_flip_horizon(self, img, anno):
        # -------------------------------------
        # 随机水平翻转
        # -------------------------------------
        if np.random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip()
            img = transform(img)
            w = img.width
            for item in anno:
                if 'item' in item:
                    landmarks = torch.tensor(anno[item]['landmarks'])
                    landmarks = landmarks.view((-1, 3))
                    landmarks[:, 0] = 2 - landmarks[:, 0]
                    landmarks = landmarks.view((landmarks.size(0) * landmarks.size(1)))
                    landmarks = landmarks.numpy().tolist()
                    anno[item]['landmarks'] = landmarks
            people_boxes = anno['people_boxes']
            people_boxes = torch.tensor(people_boxes)
            people_boxes[:, 0] = w - people_boxes[:, 0]
            people_boxes[:, 2] = w - people_boxes[:, 2]
            people_boxes = people_boxes.numpy().tolist()
            anno['people_boxes'] = people_boxes
        return img, anno

    # ------------------------------------------------------
    # 以下img皆为Tensor类型
    # ------------------------------------------------------

    def random_bright(self, img, u=32):
        # -------------------------------------
        # 随机亮度变换
        # -------------------------------------
        if np.random.random() > 0.5:
            alpha = np.random.uniform(-u, u) / 255
            img += alpha
            img = img.clamp(min=0.0, max=1.0)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5):
        # -------------------------------------
        # 随机增强对比度
        # -------------------------------------
        if np.random.random() > 0.5:
            alpha = np.random.uniform(lower, upper)
            img *= alpha
            img = img.clamp(min=0, max=1.0)
        return img

    def random_saturation(self, img, lower=0.5, upper=1.5):
        # -----------------------------------------------
        # 随机饱和度变换，针对彩色三通道图像，中间通道乘以一个值
        # -----------------------------------------------
        if np.random.random() > 0.5:
            alpha = np.random.uniform(lower, upper)
            img[1] = img[1] * alpha
            img[1] = img[1].clamp(min=0, max=1.0)
        return img

    def add_gasuss_noise(self, img, mean=0, std=0.1):
        noise = torch.normal(mean, std, img.shape)
        img += noise
        img = img.clamp(min=0, max=1.0)
        return img

    def add_salt_noise(self, img):
        noise = torch.rand(img.shape)
        alpha = np.random.random()
        img[noise[:, :, :] > alpha] = 1.0
        return img

    def add_pepper_noise(self, img):
        noise = torch.rand(img.shape)
        alpha = np.random.random()
        img[noise[:, :, :] > alpha] = 0
        return img

    def draw_img(self, img, boxes):
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='yellow', width=2)
        return img


image_path = '/home/corona/deepfooling/image'
annos_path = '/home/corona/deepfooling/annos'

aug_image_path = '/home/corona/deepfooling/aug/image'
aug_annos_path = '/home/corona/deepfooling/aug/annos'

images_name = os.listdir(image_path)
images_name = [item[:-4] for item in images_name]
aug_images_name = [item + '_1' for item in images_name]

images_path = [os.path.join(image_path, item + '.jpg') for item in images_name]
annos_path = [os.path.join(annos_path, item + '.json') for item in images_name]

aug_images_path = [os.path.join(aug_image_path, item + '_1.jpg') for item in images_name]
aug_annos_path = [os.path.join(aug_annos_path, item + '_1.json') for item in images_name]
a = ''
for item in aug_annos_path:
    a += item
    a += '\n'

# with open('a.json.txt', 'w') as f:
#     f.write(a.json)

D = DataAugmentation()

# for i in tqdm(range(len(images_path))):
#     image = Image.open(images_path[i])
#
#     with open(annos_path[i], 'r') as f:
#         anno = json.loads(f.read())
#     image, anno = D.random_flip_horizon(image, anno)
#     image = functional.pil_to_tensor(image) / 255.
#     image = D.random_bright(image)
#     image = D.random_contrast(image)
#     image = D.random_saturation(image)
#     image = np.array(functional.to_pil_image(image))
#     cv2.imwrite(aug_images_path[i], image)
#     anno = json.dumps(anno)
#     with open(aug_annos_path[i], 'w') as f:
#         f.write(anno)

import random
index = random.randint(0,800)
image = Image.open(aug_images_path[index])
with open(annos_path[index], 'r') as f:
    anno = json.loads(f.read())

boxes = anno['people_boxes']
image = D.draw_img(image, boxes)

import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()

# D = DataAugmentation()
# img1 = Image.open('C:/AllProgram/testimage/superResolution/bird_GT.bmp')
# img2 = Image.open('C:/AllProgram/testimage/superResolution/lenna.bmp')
#
# box1 = torch.Tensor([[92, 20, 261, 190]])
# box2 = torch.Tensor([[23, 18, 97, 123],
#                      [45, 65, 194, 225]])
#
# to_tensor = transforms.ToTensor()
# to_image = transforms.ToPILImage()
#
# img1, box1 = D.resize(img1, box1, 256)
# img2, box2 = D.resize(img2, box2, 256)
# img1 = to_tensor(img1)
# img2 = to_tensor(img2)
# miximg, box = D.mixup(img1, img2, box1, box2)
#
# miximg = to_image(miximg)
# D.draw_img(miximg, box)
