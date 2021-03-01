from __future__ import absolute_import
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.parse_annotations import ParseTools
import torch.nn.functional as F
from patch_config import patch_configs
from torchvision import transforms
from PIL import Image


# load yolo datasets
class ListDataset(Dataset):
    def __init__(self, txt):
        super(ListDataset, self).__init__()
        self.parser = ParseTools()
        self.configs = patch_configs['base']()
        self.img_list = self.parser.load_images_txt(txt)
        self.max_lab = 17

    def __getitem__(self, id):
        image_path = self.img_list[id]
        image_info = self.parser.load_image(image_path, mode='cv2')
        image, boxes = self.pad_and_scale(image_info['image'], image_info['boxes'])
        boxes, labels = self.pad_lab_and_boxes(boxes, image_info['labels'])

        image = transforms.ToTensor()(image)
        return image, boxes, labels

    def pad_lab_and_boxes(self, boxes, labels):
        labels = torch.tensor(labels)
        boxes = torch.tensor(boxes)
        pad_size = self.max_lab - labels.shape[0]
        if (pad_size > 0):
            # pad labels and boxes to make it have the save size
            labels = F.pad(labels, (0, pad_size), value=-1)
            boxes = F.pad(boxes, (0, 0, 0, pad_size), value=0)
        return boxes, labels

    def __len__(self):
        return len(self.img_list)

    # resize image and boxes
    def pad_and_scale(self, image, boxes):
        h = image.shape[0]
        w = image.shape[1]
        image = Image.fromarray(image)
        boxes = np.array(boxes)
        if w == h:
            padded_img = image
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(image, (int(padding)), 0)
                boxes[:, 0] = (boxes[:, 0] * w + padding) / h
                boxes[:, 2] = (boxes[:, 2] * w) / h
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(image, (0, int(padding)))
                boxes[:, 1] = (boxes[:, 1] * h + padding) / w
                boxes[:, 3] = (boxes[:, 3] * h) / w
        resize = transforms.Resize((self.configs.img_size[0], self.configs.img_size[1]))
        padded_img = resize(padded_img)
        return padded_img, boxes


if __name__ == '__main__':
    config = patch_configs['base']()
    datasets = ListDataset(config.txt_path)
    print(datasets[0])
