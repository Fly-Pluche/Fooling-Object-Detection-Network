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


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: 信噪比，Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# load datasets
class ListDataset(Dataset):
    """
    load datasets from a.json txt file which contains the path of the image file (yolo datasets' type)
    Args:
        txt: the txt file path
        number: how many pictures you want to load. default is all the image
    item:
        image: np array [w,h,3]
        boxes: [max_lab,4]
        labels: [-1,4]
    """

    def __init__(self, txt, number=None, range_=None, only_people=True):
        super(ListDataset, self).__init__()
        self.parser = ParseTools()
        self.configs = patch_configs['base']()
        if number is not None:
            self.file_list = self.parser.load_file_txt(txt)[:number]
        elif range_ is not None:
            self.file_list = self.parser.load_file_txt(txt)[range_[0]:range_[1]]
        else:
            self.file_list = self.parser.load_file_txt(txt)
        self.max_lab = 11
        self.add_noise = AddPepperNoise(0.9)
        self.only_people = only_people

    def __getitem__(self, id):
        image_path = self.file_list[id]
        image_info = self.parser.load_image(image_path, mode='numpy',
                                            root=self.configs.root_path)  # load a.json rgb type image
        boxes = np.array(image_info['boxes'])
        labels = np.array(image_info['labels'])
        if self.only_people:
            boxes = boxes[labels == 0, :]
            labels = labels[labels == 0]

        image, boxes = self.pad_and_scale(image_info['image'], boxes)
        boxes, labels = self.pad_lab_and_boxes(boxes, labels)
        # image enhance
        image = transforms.ColorJitter(brightness=0.5)(image)
        image = transforms.ColorJitter(contrast=0.5)(image)
        image = transforms.ColorJitter(hue=0.5)(image)
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
    def __init__(self, txt, number=None, name=False, range_=None):
        super(ListDatasetAnn, self).__init__(txt, number, range_)
        self.needed_classes = [1]  # the type clothes you want to select
        self.configs = patch_configs['base']()
        self.max_lab = self.configs.max_lab
        self.max_landmarks = self.configs.max_lab
        self.name = name
        self.tools = ParseTools()
        self.transform = self.get_transform()

    def __getitem__(self, id):
        anno_path = self.file_list[id]
        image_info = self.parser.parse_anno_file(anno_path=anno_path, need_classes=self.needed_classes,
                                                 root=self.configs.root_path)
        image = image_info['image']
        image_size = image_info['image_size']
        clothes_boxes = image_info['bounding_boxes']  # [x,y,x,y] uint8
        labels = image_info['people_labels']
        landmarks = image_info['landmarks']
        people_boxes = image_info['person_boxes']  # [x,y,x,y] uint 8
        landmarks = np.array(landmarks, dtype=np.float)
        landmarks[:, :, 0] = landmarks[:, :, 0] / image_size[0]
        landmarks[:, :, 1] = landmarks[:, :, 1] / image_size[1]
        image, clothes_boxes, people_boxes, landmarks = self.pad_and_scale_(image, clothes_boxes, people_boxes,
                                                                            landmarks)
        # instead clothes labels with people labels

        clothes_boxes, people_boxes, labels = self.pad_lab_and_boxes_(clothes_boxes, people_boxes, labels)
        landmarks = self.pad_landmarks(landmarks)
        image = self.transform(image)
        image = functional.pil_to_tensor(image) / 255.
        # image: [3,w,h] boxes: [max_pad, 4] (x,y,w,h) labels: [max_pad] landmarks: [max_landmarks,25,3]
        segmentations = self.landmarks2masks(landmarks)
        segmentations = torch.from_numpy(segmentations)
        # get clothes' mask
        segmentations = segmentations.unsqueeze(1)
        segmentations = segmentations.expand(-1, 3, -1, -1)
        segmentations = segmentations * image
        if self.name:
            return image, clothes_boxes, labels, landmarks, segmentations, image_info['image_name']
        return image, clothes_boxes, people_boxes, labels, landmarks, segmentations

    def get_transform(self):
        # 随机改变图像的亮度
        brightness_change = transforms.ColorJitter(brightness=0.5)
        # 随机改变图像的色调
        hue_change = transforms.ColorJitter(hue=0.5)
        # 随机改变图像的对比度
        contrast_change = transforms.ColorJitter(contrast=0.5)
        transform = transforms.Compose([
            brightness_change,
            hue_change,
            contrast_change,
        ])
        return transform

    def pad_lab_and_boxes_(self, clothes_boxes, people_boxes, labels):
        labels = torch.from_numpy(np.array(labels))
        clothes_boxes = torch.from_numpy(np.array(clothes_boxes))
        people_boxes = torch.from_numpy(np.array(people_boxes))
        clothes_pad_size = self.max_lab - clothes_boxes.size(0)
        people_pad_size = self.max_lab - people_boxes.size(0)
        if (people_pad_size > 0):
            # pad labels and boxes to make it have the save size
            labels = F.pad(labels, (0, people_pad_size), value=-1)
            people_boxes = F.pad(people_boxes, (0, 0, 0, people_pad_size), value=0)
        if (clothes_pad_size > 0):
            clothes_boxes = F.pad(clothes_boxes, (0, 0, 0, clothes_pad_size), value=0)

        return clothes_boxes, people_boxes, labels

    def landmarks2masks(self, landmarks):
        """
        a.json batch of landmarks to a.json batch of masks
        Args:
            landmarks: [batch size, max_landmarks, 3]
        """
        masks_batch = []
        landmarks_scaled = np.copy(landmarks)
        landmarks_scaled = landmarks_scaled * self.configs.img_size_big[0]
        polygons_batch = landmarks_scaled[:, :, 0: 2]  # [batch size, max_landmarks, )]
        bg_mask = np.zeros(self.configs.img_size_big, dtype=np.uint8)
        for polygons in polygons_batch:
            polygons = np.array(polygons, np.int32)
            if np.sum(polygons) != 0:
                masks = cv2.fillPoly(bg_mask, [polygons], color=1)
                masks_batch.append(np.asarray(Image.fromarray(masks)))
            else:
                masks_batch.append(bg_mask)
        masks_batch = np.array(masks_batch, np.float)
        return masks_batch

    def pad_and_scale_(self, image, clothes_boxes, people_boxes, landmarks):
        h = image.shape[0]
        w = image.shape[1]
        image = Image.fromarray(image)
        clothes_boxes = np.array(clothes_boxes)
        people_boxes = np.array(people_boxes)
        landmarks = np.array(landmarks)
        people_boxes = self.tools.xyxy2xywh_batch((w, h), people_boxes)

        if w == h:
            padded_img = image
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(image, (int(padding), 0))
                # clothes boxes: xywh, people boxes : xyxy
                clothes_boxes[:, 0] = (clothes_boxes[:, 0] * w + padding) / h
                clothes_boxes[:, 2] = (clothes_boxes[:, 2] * w) / h
                people_boxes[:, 0] = (people_boxes[:, 0] * w + padding) / h
                people_boxes[:, 2] = (people_boxes[:, 2] * w) / h
                landmarks[:, :, 0] = ((landmarks[:, :, 0] * w) + padding) / h

            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(image, (0, int(padding)))
                clothes_boxes[:, 1] = (clothes_boxes[:, 1] * h + padding) / w
                clothes_boxes[:, 3] = (clothes_boxes[:, 3] * h) / w
                people_boxes[:, 1] = (people_boxes[:, 1] * h + padding) / w
                people_boxes[:, 3] = (people_boxes[:, 3] * h) / w
                landmarks[:, :, 1] = (landmarks[:, :, 1] * h + padding) / w

        resize = transforms.Resize((self.configs.img_size_big[0], self.configs.img_size_big[1]))
        padded_img = resize(padded_img)
        return padded_img, clothes_boxes, people_boxes, landmarks

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
    # plt.pytorch_imshow(image)
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
