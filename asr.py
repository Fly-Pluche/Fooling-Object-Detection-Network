import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from load_data import ListDatasetAnn, ListDataset
from torch.utils.data import DataLoader
from models import Yolov3
from patch_config import patch_configs
import os
from patch import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class BaseASR(nn.Module):
    def __init__(self, image_size, use_deformation=True):
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
        self.use_deformation = use_deformation
        self.is_register = False
        if use_deformation:
            self.patch_transformer = PatchTransformerPro().cuda()
            self.patch_applier = PatchApplierPro().cuda()
        else:
            self.patch_transformer = PatchTransformer().cuda()
            self.patch_applier = PatchApplier().cuda()

    def register_dataset(self, model, data_loader):
        """
        register model and data loader
        Args:
            model: any model you need to calculate asr
            data_loader: a Pytorch data loader
        """
        if not self.is_register:
            self.model = model
            self.data_loader = data_loader
            self.is_register = True
        # else:
        #     self.is_register = True
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

    def inference_on_dataset(self, adv_patch=None, clean=False):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.forward` accurately.
        The model will be used in eval mode.
        """
        predicted_dicts = []
        dataset_dicts = []
        total_boxes_number = 0
        with torch.no_grad():
            for id, item in enumerate(
                    tqdm(self.data_loader)):
                if self.use_deformation:
                    image_batch, clothes_boxes_batch, _, _, landmarks_batch, segmentations_batch = item
                    clothes_boxes_batch = clothes_boxes_batch.cuda()
                    segmentations_batch = segmentations_batch.cuda()
                    landmarks_batch = landmarks_batch.cuda()
                else:
                    image_batch, people_boxes, labels_batch = item
                    people_boxes = people_boxes.cuda()
                    labels_batch = labels_batch.cuda()
                image_batch = image_batch.cuda()
                if not clean:
                    adv_patch = adv_patch.cuda()
                    if self.use_deformation:
                        adv_batch_t, adv_batch_mask_t = self.patch_transformer(adv_patch, clothes_boxes_batch,
                                                                               segmentations_batch, landmarks_batch,
                                                                               image_batch)
                        p_img_batch = self.patch_applier(image_batch, adv_batch_t, adv_batch_mask_t)
                    else:
                        adv_batch_t = self.patch_transformer(adv_patch, people_boxes, labels_batch.clone())
                        p_img_batch = self.patch_applier(image_batch, adv_batch_t)
                else:
                    p_img_batch = image_batch
                image_batch = F.interpolate(p_img_batch, (self.image_size[0], self.image_size[1]))
                # images = torch.unbind(image_batch, dim=0)
                for idx in range(image_batch.size(0)):
                    image = image_batch[idx]
                    if self.model.model_name == 'Yolov3':
                        outputs = self.model(image, nms=True)
                    else:
                        outputs = self.model(image)
                    outputs = outputs['instances']
                    boxes = outputs.pred_boxes.tensor
                    boxes = boxes[outputs.pred_classes == self.model.people_index, :]
                    # if boxes.device != 'cpu':
                    #     boxes = boxes.cpu()
                    predicted_dicts.append({
                        'bbox': boxes.cpu().numpy(),
                    })
                    # --------------------------
                    boxes = people_boxes[idx]
                    labels = labels_batch[idx].view(-1)
                    # only load people boxes
                    boxes = boxes[labels == 0, :]
                    # remove padded boxes
                    box_sum = torch.sum(boxes, dim=1)
                    boxes = boxes[box_sum != 0, :]
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
                        'bbox': boxes.cpu().numpy(),
                    })
        self.predicted_dicts = predicted_dicts
        self.dataset_dicts = dataset_dicts
        self.total_bbox_number = total_boxes_number

    def calculate_asr(self):
        pass


class ObjectVanishingASR(BaseASR):
    def __init__(self, image_size, use_deformation=True):
        super(ObjectVanishingASR, self).__init__(image_size, use_deformation)

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
        ListDataset(config.coco_val_txt, range_=[0, 100]),
        num_workers=4,
        batch_size=config.batch_size,
        shuffle=False
    )
    print(config.img_size)
    import cv2

    adv_patch = cv2.imread('./patches/0.jpg')
    adv_patch = functional.to_tensor(adv_patch)
    base = ObjectVanishingASR(config.img_size, use_deformation=False)
    model = Yolov3(model_path='model_data/yolo_weight_voc_608.pth', image_size=608)
    model.set_image_size(config.img_size[0])
    base.register_dataset(model, test_data)
    base.inference_on_dataset(adv_patch, clean=True)
    asr = base.calculate_asr()
    print(asr)
