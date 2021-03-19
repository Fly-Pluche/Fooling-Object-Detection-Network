import torch
import copy
import time
import logging
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from patch import PatchTransformerPro, PatchApplierPro
from patch_config import patch_configs
from load_data import ListDatasetAnn
from models import RetinaNet, MaskRCNN, FasterRCNN
from torchvision.transforms import functional

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MaxProbExtractor(nn.Module):
    """
    get the max score in a batch of images
    """

    def __init__(self):
        super(MaxProbExtractor, self).__init__()

    def forward(self, model, batch_image):
        images = torch.unbind(batch_image, 0)
        max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.scores
            people_scores = scores[pred_classes == 0]  # select people predict score
            if len(people_scores) != 0:
                max_prob = torch.max(people_scores)
                max_prob_t[i] = max_prob
        return max_prob_t


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchEvaluator(nn.Module):
    def __init__(self, model, data_loader):
        super(PatchEvaluator, self).__init__()
        self.p_config = patch_configs['base']()
        self.calculator = CalculateAP()
        self.data_loader = None
        self.model = None
        self.predicts = None
        self.ground_truths = None
        self.image_sizes = None
        self.register_dataset(model, data_loader)
        self.patch_transformer = PatchTransformerPro().cuda()
        self.patch_applier = PatchApplierPro().cuda()
        self.class_id = 0  # the class you want to calculate ap

    def register_dataset(self, model, data_loader):
        self.data_loader = data_loader
        self.model = model
        self.ground_truths = self.get_people_dicts(data_loader)

    # get people dicts from pytorch data loader
    def get_people_dicts(self, data_loader):
        dataset_dicts = []
        for idx, (image_batch, _, people_boxes, labels_batch, _, _) in enumerate(
                tqdm(data_loader, ascii=True, desc='load people\'s boxes information')):
            for id in range(image_batch.size(0)):
                boxes = people_boxes[id]
                labels = labels_batch[id].view(-1)
                boxes = boxes[labels == 0, :]
                s = image_batch.size()
                w = boxes[:, 2] * s[2]  # w
                h = boxes[:, 3] * s[3]  # h
                boxes[:, 0] = boxes[:, 0] * s[2] - w / 2
                boxes[:, 1] = boxes[:, 1] * s[3] - h / 2
                boxes[:, 2] = boxes[:, 0] + w
                boxes[:, 3] = boxes[:, 1] + h
                det = np.zeros(boxes.size(0))
                dataset_dicts.append({
                    'bbox': boxes.numpy(),
                    'det': det
                })
        dataset_dicts = np.array(dataset_dicts)
        return dataset_dicts

    def forward(self, adv_patch, threshold=0.5):
        predicts = self.inference_on_dataset(adv_patch)
        ground_truths = copy.deepcopy(self.ground_truths)
        ap = self.calculator.ap(self.class_id, predicts, ground_truths, self.image_sizes, threshold=threshold)
        return ap

    def inference_on_dataset(self, adv_patch):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.forward` accurately.
        The model will be used in eval mode.
        """
        images_ids_ = []
        confidences_ = []
        BB = []
        with torch.no_grad():
            for id, (
                    image_batch, clothes_boxes_batch, _, _, landmarks_batch, segmentations_batch) in enumerate(
                tqdm(self.data_loader)):
                image_batch = image_batch.cuda()
                clothes_boxes_batch = clothes_boxes_batch.cuda()
                landmarks_batch = landmarks_batch.cuda()
                segmentations_batch = segmentations_batch.cuda()
                adv_patch = adv_patch.cuda()
                adv_batch_t, adv_batch_mask_t = self.patch_transformer(adv_patch,
                                                                       clothes_boxes_batch,
                                                                       segmentations_batch,
                                                                       landmarks_batch,
                                                                       image_batch)
                p_img_batch = self.patch_applier(image_batch, adv_batch_t, adv_batch_mask_t)
                p_img_batch = F.interpolate(p_img_batch, (self.p_config.img_size[1], self.p_config.img_size[0]))
                images = torch.unbind(p_img_batch, dim=0)
                for idx, image in enumerate(images):
                    outputs = self.model(image)
                    # a = model.visual_instance_predictions(image, outputs)
                    # plt.imshow(a)
                    # plt.show()
                    outputs = outputs['instances']
                    boxes = outputs.pred_boxes
                    scores = outputs.scores
                    classes = outputs.pred_classes

                    # select the class we want
                    boxes = boxes[classes == self.class_id].tensor
                    scores = scores[classes == self.class_id]
                    classes = classes[classes == self.class_id]
                    classes = classes[scores >= 0.5]
                    boxes = boxes[scores >= 0.5, :]
                    scores = scores[scores >= 0.5]

                    images_ids = torch.cuda.FloatTensor(boxes.size(0), 1).fill_(idx + id * self.p_config.batch_size)

                    scores = scores.unsqueeze(-1)

                    classes = classes.unsqueeze(-1)
                    classes = torch.cat((images_ids, classes), dim=1)
                    boxes = torch.cat((images_ids, boxes), dim=1)
                    images_ids_.append(images_ids)
                    BB.append(boxes)
                    confidences_.append(scores)

        images_ids_ = torch.cat(images_ids_, dim=0).squeeze(-1)

        BB = torch.cat(BB, dim=0)
        confidences_ = torch.cat(confidences_, dim=0).squeeze(-1)
        return (images_ids_, confidences_, BB)


class CalculateAP:
    def __init__(self):
        pass

    def ap(self, class_id, predicts, ground_truths, image_sizes, threshold=0.5):
        # extract gt objects for this class
        image_ids = predicts[0].cpu().numpy()
        confidence = predicts[1].cpu().numpy()
        BB = predicts[2].cpu().numpy()

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = ground_truths[int(image_ids[d])]
            bb = BB[d, 1:5].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                if ovmax > threshold:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(nd)

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric=True)
        return ap

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 计算PR曲线向下包围的面积
        return ap

    # def ap(self, class_id, predicts, ground_truths, image_sizes, threshold=0.5):
    #     """
    #     calculate ap of one class
    #     class id: the class which you want to calculate ap
    #     predicts: a list contains numpy arrays  np array format [n,6]
    #     ground_truths: box and label information read from datasets
    #     image_sizes: [(width,height),(width,height) ... ] recording the images' width and height
    #     threshold:  box's iou > threshold means this box is a right box
    #     """
    #     precisions = []
    #     recalls = []
    #     # calculate predictions and recalls
    #     for predict, ground_truth, image_size in zip(predicts, ground_truths, image_sizes):
    #         precision, recall = self.precision_recall(class_id, np.array(predict), np.array(ground_truth), image_size,
    #                                                   threshold)
    #         precisions.append(precision)
    #         recalls.append(recall)
    #     # correct AP calculation
    #     # first append sentinel values at the end
    #     mrec = np.concatenate(([0.], recalls, [1.]))
    #     mpre = np.concatenate(([0.], precisions, [0.]))
    #
    #     # compute the precision envelope
    #     for i in range(mpre.size - 1, 0, -1):
    #         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    #     # to calculate area under PR curve, look for points
    #     # where X axis (recall) changes value
    #     i = np.where(mrec[1:] != mrec[:-1])[0]
    #     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    #     return ap

    def precision_recall(self, class_id, predict, ground_truth, image_size, threshold=0.5, predict_type='xyxy',
                         ground_truth_type='xywh'):
        """
        calculate precision and recall (two numpy arrays)
        class id: the class which you want to calculate ap
        predict: a np array [n, 6]
                 n_: the number of each image's boxes/labels
                 the dim 2 include:
                    label: the id of this class
                    confidence: the confidences of each box
                    box: [x,y,w,h] between 0 and 1
        ground truth: a np array with the same size of the predict
        image_sizes: (width,height)  recording the image's width and height
        threshold: box's iou > threshold means this box is a right box
        predict_type: the boxes's type, if your boxes'
        """
        width, height = image_size[0], image_size[1]
        p = predict
        p = p[p[:, 0] == class_id]
        p = p[p[:, 1] >= 0.3]  # remove the low conf boxes, NMS
        ground_truth = ground_truth[ground_truth[:, 0] == class_id]
        # sorted by confidence
        conf_sort_id = np.argsort(-p[:, 1])
        if len(conf_sort_id) == 0:
            return 0, 0
        p = np.delete(p, 1, axis=1)
        # go down dets and mark TPs and Fps
        # nd = ground_truth.size(0)
        # tp = np.zeros(nd)
        # fp = np.zeros(nd)

        # xywh to xyxy
        if predict_type == 'xywh':
            p = self.__xywh2xyxy(p, height, width)
        if ground_truth_type == 'xywh':
            ground_truth = self.__xywh2xyxy(ground_truth, height, width)
        is_select = np.zeros(ground_truth.shape[0])
        # [n,6] 6: label x_min, y_min, x_max, y_max, is_select
        ground_truth = np.c_[ground_truth, is_select]
        for i in conf_sort_id:
            gt_ = np.copy(ground_truth)
            gt_ = gt_[gt_[:, -1] == 0]
            # calculate boxes' iou
            p_area = abs((p[i][3] - p[i][1]) * (p[i][4] - p[i][2]))
            gt_areas = np.abs((gt_[:, 3] - gt_[:, 1]) * (gt_[:, 4] - gt_[:, 2]))
            x_min = np.max((gt_[:, 1], np.ones_like(gt_[:, 1]) * p[i][1]), axis=0, keepdims=True)
            y_min = np.max((gt_[:, 2], np.ones_like(gt_[:, 1]) * p[i][2]), axis=0, keepdims=True)
            x_max = np.min((gt_[:, 3], np.ones_like(gt_[:, 1]) * p[i][3]), axis=0, keepdims=True)
            y_max = np.min((gt_[:, 4], np.ones_like(gt_[:, 1]) * p[i][4]), axis=0, keepdims=True)

            iou_w = x_max - x_min
            iou_h = y_max - y_min
            overlap = iou_w * iou_h
            min_area = np.copy(gt_areas)
            min_area[min_area > p_area] = p_area
            overlap[overlap > min_area] = 0

            # overlap[overlap < 0] = 0

            union = p_area + gt_areas - overlap
            iou = overlap / union

            idx = np.argsort(iou[iou > threshold])
            if len(idx) > 0:
                ground_truth[idx[-1], -1] = 1

        TP = ground_truth[ground_truth[:, -1] == 1].shape[0]
        precision = TP / (len(conf_sort_id))
        recall = TP / ground_truth.shape[0]
        return precision, recall

    def __xywh2xyxy(self, box, height, width):
        box[:, 1] = box[:, 1] * width
        box[:, 2] = box[:, 2] * height
        box[:, 3] = box[:, 3] * width
        box[:, 4] = box[:, 4] * height
        box[:, 1] = box[:, 1] - box[:, 3] / 2
        box[:, 3] = box[:, 1] + box[:, 3]
        box[:, 2] = box[:, 2] - box[:, 4] / 2
        box[:, 4] = box[:, 2] + box[:, 4]
        box = np.array(box, dtype=np.int)
        return box


if __name__ == '__main__':
    patch_config = patch_configs['base']()
    # load train datasets
    datasets = ListDatasetAnn(patch_config.deepfashion_txt, number=100)
    data = DataLoader(
        datasets,
        batch_size=patch_config.batch_size,
        num_workers=2,
        shuffle=False,
    )
    model = RetinaNet()
    attack_evaluator = PatchEvaluator(model, data)
    # print(attack_evaluator.ground_truths)
    patch = Image.open("/home/corona/attack/Fooling-Object-Detection-Network/patches/patch2.png")
    patch = patch.resize((500, 500))
    adv_patch = torch.cuda.FloatTensor(transforms.PILToTensor()(patch).cuda() / 255)

    adv_patch = adv_patch.cuda()
    ap = attack_evaluator(adv_patch)
    print(ap)
