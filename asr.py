import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from load_data import ListDatasetAnn
from torch.utils.data import DataLoader
from models import Yolov3
from patch_config import patch_configs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class BaseASR(nn.Module):
    def __init__(self, image_size):
        """
        Base ASR
        Args:
            image_size: the width and height of image: [w,h]
        """
        super(BaseASR, self).__init__()
        self.model = None
        self.data_loader = None
        self.ground_truths = None
        self.total_bbox_number = None
        self.dataset_dicts = None
        self.predicted_dicts = None
        self.image_size = image_size

    def register_dataset(self, model, data_loader):
        """
        register model and data loader
        Args:
            model: any model you need to calculate asr
            data_loader: a Pytorch data loader
        """
        self.model = model
        self.data_loader = data_loader
        # self.dataset_dicts = self.get_people_dicts(data_loader)

    def get_people_dicts(self, data_loader):
        """
        get people dicts from a Pytorch data loader
        tips: you can change this function according to your own data loader
        Args:
            data_loader: a Pytorch data loader
        """
        dataset_dicts = []
        total_boxes_number = 0
        for idx, (image_batch, _, people_boxes, labels_batch, _, _) in enumerate(
                tqdm(data_loader, ascii=True, desc='load people\'s boxes information')):
            for id in range(image_batch.size(0)):
                if id == idx == 1:
                    self.image1 = image_batch[0]
                boxes = people_boxes[id]
                labels = labels_batch[id].view(-1)
                # only load people boxes
                boxes = boxes[labels == 0, :]
                s = image_batch.size()
                w = boxes[:, 2] * s[2]  # w
                h = boxes[:, 3] * s[3]  # h
                # xywh ==> xyxy
                boxes[:, 0] = boxes[:, 0] * s[2] - w / 2
                boxes[:, 1] = boxes[:, 1] * s[3] - h / 2
                boxes[:, 2] = boxes[:, 0] + w
                boxes[:, 3] = boxes[:, 1] + h
                total_boxes_number += int(boxes.size(0))
                dataset_dicts.append({
                    'bbox': boxes.numpy(),
                })
        dataset_dicts = np.array(dataset_dicts)
        self.total_bbox_number = total_boxes_number
        return dataset_dicts

    def inference_on_dataset(self):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.forward` accurately.
        The model will be used in eval mode.
        """
        predicted_dicts = []
        dataset_dicts = []
        total_boxes_number = 0
        with torch.no_grad():
            for id, (
                    image_batch, clothes_boxes_batch, people_boxes, labels_batch, landmarks_batch,
                    segmentations_batch) in enumerate(
                tqdm(self.data_loader)):
                image_batch = image_batch.cuda()
                image_batch = F.interpolate(image_batch, (self.image_size[0], self.image_size[1]))
                # images = torch.unbind(image_batch, dim=0)
                for idx in range(image_batch.size(0)):
                    image = image_batch[idx]
                    if self.model.model_name == 'Yolov3':
                        outputs = self.model(image, nms=True)
                    else:
                        outputs = self.model(image)
                    outputs = outputs['instances']
                    boxes = outputs.pred_boxes.tensor
                    if boxes.device != 'cpu':
                        boxes = boxes.cpu()
                    predicted_dicts.append({
                        'bbox': boxes.numpy(),
                    })
                    # --------------------------
                    boxes = people_boxes[idx]
                    labels = labels_batch[idx].view(-1)
                    # only load people boxes
                    boxes = boxes[labels == 0, :]
                    s = image_batch.size()
                    w = boxes[:, 2] * s[2]  # w
                    h = boxes[:, 3] * s[3]  # h
                    # xywh ==> xyxy
                    boxes[:, 0] = boxes[:, 0] * s[2] - w / 2
                    boxes[:, 1] = boxes[:, 1] * s[3] - h / 2
                    boxes[:, 2] = boxes[:, 0] + w
                    boxes[:, 3] = boxes[:, 1] + h
                    total_boxes_number += int(boxes.size(0))
                    dataset_dicts.append({
                        'bbox': boxes.numpy(),
                    })
        self.predicted_dicts = predicted_dicts
        self.dataset_dicts = dataset_dicts
        self.total_bbox_number = total_boxes_number

    def calculate_asr(self):
        pass


class ObjectVanishingASR(BaseASR):
    def __init__(self, image_size):
        super(ObjectVanishingASR, self).__init__(image_size)

    def calculate_asr(self):
        success_boxes_number = 0
        predicted_dicts = self.predicted_dicts
        for i, predicted_dict in enumerate(predicted_dicts):
            gt_boxes = self.dataset_dicts[i]['bbox']
            pre_boxes = predicted_dict['bbox']
            for j in range(gt_boxes.shape[0]):
                if len(pre_boxes) == 0:
                    success_boxes_number += 1
                    continue
                ixmin = np.maximum(pre_boxes[:, 0], gt_boxes[j, 0])
                iymin = np.maximum(pre_boxes[:, 1], gt_boxes[j, 1])
                ixmax = np.minimum(pre_boxes[:, 2], gt_boxes[j, 2])
                iymax = np.minimum(pre_boxes[:, 3], gt_boxes[j, 3])
                iw = np.maximum(ixmax - ixmin, 0)
                ih = np.maximum(iymax - iymin, 0)
                inters = iw * ih
                union = ((gt_boxes[j, 2] - gt_boxes[j, 0]) * (gt_boxes[j, 3] - gt_boxes[j, 1]) +
                         (pre_boxes[:, 2] - pre_boxes[:, 0]) *
                         (pre_boxes[:, 3] - pre_boxes[:, 1]) - inters)
                iou = np.max(inters / union)
                print(iou)
                if iou < 0.5:
                    success_boxes_number += 1

        # calculate ASR
        return success_boxes_number / self.total_bbox_number


class ObjectFabricationASR(BaseASR):
    def __init__(self, image_size):
        super(ObjectFabricationASR, self).__init__(image_size)


class ObjectMislabelingASR(BaseASR):
    def __init__(self, image_size):
        super(ObjectMislabelingASR, self).__init__(image_size)


if __name__ == '__main__':
    config = patch_configs['base']()
    test_data = DataLoader(
        ListDatasetAnn(config.deepfooling_txt, range_=[0, 80]),
        num_workers=4,
        batch_size=config.batch_size,
        shuffle=False
    )
    print(config.img_size)

    base = ObjectVanishingASR(config.img_size)
    model = Yolov3(model_path='net/yolov3/logs/Epoch10-Total_Loss11.4261-Val_Loss10.1546.pth', image_size=608)
    model.set_image_size(config.img_size[0])
    base.register_dataset(model, test_data)
    base.inference_on_dataset()
    asr = base.calculate_asr()
    print(asr)
