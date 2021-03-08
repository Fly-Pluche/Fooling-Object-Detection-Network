from __future__ import absolute_import
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.parse_annotations import ParseTools
import torch.nn.functional as F
from patch_config import patch_configs
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)


# load datasets
class ListDataset(Dataset):
    """
    load datasets from a txt file which contains the path of the image file (yolo datasets' type)
    Args:
        txt: the txt file path
        number: how many pictures you want to load. default is all the image
    item:
        image: np array [w,h,3]
        boxes: [max_lab,4]
        labels: [-1,4]
    """

    def __init__(self, txt, number=None):
        super(ListDataset, self).__init__()
        self.parser = ParseTools()
        self.configs = patch_configs['base']()
        if number is None:
            self.file_list = self.parser.load_file_txt(txt)
        else:
            self.file_list = self.parser.load_file_txt(txt)[:number]
        self.max_lab = 21

    def __getitem__(self, id):
        image_path = self.file_list[id]
        image_info = self.parser.load_image(image_path, mode='numpy')  # load a rgb type image
        # boxes: [x,y,w,h]
        image, boxes = self.pad_and_scale(image_info['image'], image_info['boxes'])
        boxes, labels = self.pad_lab_and_boxes(boxes, image_info['labels'])
        image = transforms.ToTensor()(image)
        return image, boxes, labels

    def pad_lab_and_boxes(self, boxes, labels):
        labels = torch.from_numpy(np.array(labels))
        boxes = torch.from_numpy(np.array(boxes))
        pad_size = self.max_lab - labels.shape[0]
        if (pad_size > 0):
            # pad labels and boxes to make it have the save size
            labels = F.pad(labels, (0, pad_size), value=-1)
            boxes = F.pad(boxes, (0, 0, 0, pad_size), value=0)
        return boxes, labels

    def __len__(self):
        return len(self.file_list)

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
                padded_img.paste(image, (int(padding), 0))
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


class ListDatasetAnn(ListDataset):
    def __init__(self, txt, number=None):
        super(ListDatasetAnn, self).__init__(txt, number)
        self.needed_classes = [1]
        self.max_lab = 2
        self.max_landmarks = 2

    def __getitem__(self, id):
        anno_path = self.file_list[id]
        image_info = self.parser.parse_anno_file(anno_path=anno_path, need_classes=self.needed_classes)
        image = image_info['image']
        image_size = image_info['image_size']
        boxes = image_info['bounding_boxes']  # [x,y,x,y] uint8
        labels = image_info['category_ids']
        landmarks = image_info['landmarks']
        landmarks = np.array(landmarks,dtype=np.float)
        landmarks[:, :, 0] = landmarks[:, :, 0] / image_size[0]
        landmarks[:, :, 1] = landmarks[:, :, 1] / image_size[1]
        image, boxes, landmarks = self.pad_and_scale_(image, boxes, landmarks)
        boxes, labels = self.pad_lab_and_boxes(boxes, labels)
        landmarks = self.pad_landmarks(landmarks)
        image = functional.pil_to_tensor(image)
        # image: [3,w,h] boxes: [max_pad, 4] (x,y,w,h) labels: [max_pad] landmarks: [max_landmarks,25,3]
        return image, boxes, labels, landmarks

    def pad_and_scale_(self, image, boxes, landmarks):
        h = image.shape[0]
        w = image.shape[1]
        image = Image.fromarray(image)
        boxes = np.array(boxes)
        landmarks = np.array(landmarks)
        if w == h:
            padded_img = image
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(image, (int(padding), 0))
                # boxes: xywh
                boxes[:, 0] = (boxes[:, 0] * w + padding) / h
                boxes[:, 2] = (boxes[:, 2] * w) / h
                landmarks[:, :, 0] = ((landmarks[:, :, 0] * w) + padding) / h

            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(image, (0, int(padding)))
                boxes[:, 1] = (boxes[:, 1] * h + padding) / w
                boxes[:, 3] = (boxes[:, 3] * h) / w
                landmarks[:, :, 1] = (landmarks[:, :, 1] * h + padding) / w

        resize = transforms.Resize((self.configs.img_size_big[0], self.configs.img_size_big[1]))
        padded_img = resize(padded_img)
        return padded_img, boxes, landmarks

    def pad_landmarks(self, landmarks):
        """
        Make the data format of each batch the same
        Args:
            landmarks: [25,3]
        """
        landmarks = torch.from_numpy(landmarks)
        pad_size = self.max_landmarks - landmarks.shape[0]

        if (pad_size > 0):
            # pad labels and boxes to make it have the save size
            landmarks = F.pad(landmarks, (0, 0, 0, 0, 0, pad_size), value=0)

        return landmarks


def load_test_data_loader(txt, number=10):
    """
    load test data loader
    Args:
        txt: the txt file of the datasets (datasets format is the yolo datasets format)
        number: how many images you want to load
    """
    dataset = ListDataset(txt, number=number)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=True,
    )
    return data_loader


if __name__ == '__main__':
    config = patch_configs['base']()
    da = ListDatasetAnn(config.deepfashion_txt, 10)
    da = iter(da)
    next(da)
    next(da)
    next(da)
    # datasets = load_test_data_loader(config.txt_path)
    # datasets = iter(datasets)
    # image = next(datasets)[0][0]
    # image = np.array(functional.to_pil_image(image))
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(image)
    # plt.show()
"""
[237, 94, 1, 
90, 93, 2, 
162, 131, 2, 
268, 153, 2, 
336, 118, 2, 
292, 61, 2, 
9, 131, 2,
 0, 0, 0,
  5, 325, 2,
   118, 328, 2,
    116, 303, 2,
     118, 276, 2, 118, 312, 2, 137, 442, 2, 158, 562, 1, 322, 554, 2, 418, 486, 2, 403, 366, 2, 397, 254, 2, 397, 185, 1, 397, 206, 1, 396, 229, 1, 448, 191, 2, 424, 124, 2, 379, 60, 2]
"""
