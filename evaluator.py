import torch
import copy
import time
import logging
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from patch import PatchTransformerPro, PatchApplierPro, PatchApplier, PatchTransformer
from utils.parse_annotations import ParseTools
from patch_config import patch_configs
from load_data import ListDatasetAnn
from models import *
from models import RetinaNet, MaskRCNN, FasterRCNN
from PIL import Image
from torchvision.transforms import functional
import random
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


class MaxExtractor(nn.Module):
    """
    get the max score and self information of the max iou in a batch of images
    """

    def __init__(self):
        super(MaxExtractor, self).__init__()
        self.config = patch_configs['base']()
        self.tools = ParseTools()

    def forward(self, model, batch_image, people_boxes):
        """
        Args:
            model: the model used to predict
            batch_image: a batch of images [batch size, 3, width, height]
            people_boxes: a batch of boxes [batch size,lab number,4]
        """
        union_image = torchvision.utils.make_grid(batch_image, padding=0)
        # resize union image
        theta = torch.tensor([
            [1, 0, 0.],
            [0, 1, 0.]
        ], dtype=torch.float).cuda()
        N, C, W, H = union_image.unsqueeze(0).size()
        size = torch.Size((N, C, W // 4, H // 4))
        grid = F.affine_grid(theta.unsqueeze(0), size)
        output = F.grid_sample(union_image.unsqueeze(0), grid)
        union_image = output[0]
        # plt.imshow(np.array(functional.to_pil_image(union_image)))
        # plt.show()
        # union_image = functional.resize()
        images = torch.unbind(batch_image, 0)
        people_boxes = self.tools.xywh2xyxy_batch_torch(people_boxes, self.config.img_size)
        max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        max_iou_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.scores
            boxes = output.pred_boxes.tensor
            people_scores = scores[pred_classes == 0]  # select people predict score
            boxes = boxes[pred_classes == 0]
            iou_max = torch.tensor(0., device='cuda')
            gt_boxes = people_boxes[i]
            mask_gt_boxes = torch.sum(gt_boxes, dim=-1)
            gt_boxes = gt_boxes[mask_gt_boxes != 0]
            for j in range(boxes.size(0)):
                ixmin = torch.maximum(gt_boxes[:, 0], boxes[j, 0])
                iymin = torch.maximum(gt_boxes[:, 1], boxes[j, 1])
                ixmax = torch.minimum(gt_boxes[:, 2], boxes[j, 2])
                iymax = torch.minimum(gt_boxes[:, 3], boxes[j, 3])
                iw = torch.maximum(ixmax - ixmin, torch.tensor(0., device='cuda'))
                ih = torch.maximum(iymax - iymin, torch.tensor(0., device='cuda'))
                inters = iw * ih

                # union
                uni = ((boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]) +
                       (gt_boxes[:, 2] - gt_boxes[:, 0]) *
                       (gt_boxes[:, 3] - gt_boxes[:, 1]) - inters)

                overlaps = inters / uni
                overlaps = torch.max(overlaps)
                iou_max = torch.max(iou_max, overlaps)

            if len(people_scores) != 0:
                max_prob = torch.max(people_scores)
                max_prob = max_prob + 1
                max_prob = torch.sqrt(max_prob)
                max_prob_t[i] = max_prob
                max_iou_t[i] = iou_max
        union_detect = model(union_image)['instances']
        labels = union_detect.pred_classes
        max_prob_t = union_detect[labels == 0].scores
        return max_prob_t, max_iou_t


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


class UnionDetector(nn.Module):
    def __init__(self):
        super(UnionDetector, self).__init__()
        self.config = patch_configs['base']()

    def forward(self, model, image_batch, p_image_batch, people_boxes):
        """
        Args:
            image_batch: [batch size,3,w,h]
            p_image_batch: [batch size,3,w,h]
            people_boxes: []
        """
        # stitching a square picture
        w = int(self.config.batch_size)
        # random choose 2 normal images and 2 adversarial images
        image_index1 = [i for i in range(0, w)]
        random.shuffle(image_index1)
        image_index2 = [i for i in range(w, w * 2)]
        random.shuffle(image_index2)
        index_ = int(w / 2)
        if index_ < 2:
            index_ = 2
        image_index = image_index1[:index_] + image_index2[:index_]
        random.shuffle(image_index)
        image_index = torch.tensor(image_index, device='cuda')
        # choose image and cat the images according the image index
        images = torch.cat((image_batch, p_image_batch), dim=0)  # [2 * batch size,3,h,w]
        images = images[image_index]
        union_image = torchvision.utils.make_grid(images, nrow=int(w ** (1 / 2)), padding=0)
        # plt.imshow(np.array(functional.to_pil_image(union_image)))
        # plt.show()

        # adjust boxes coordinates
        people_boxes = torch.cat((people_boxes, people_boxes), dim=0)  # [4,3,4] => [8,3,4]
        people_boxes = people_boxes[image_index]  # => [4,3,4]
        s = people_boxes.size()
        people_boxes = people_boxes.reshape((s[0] * s[1], s[2]))
        # people_boxes = torch.unbind(people_boxes, dim=0)
        # people_boxes = torch.cat(people_boxes, dim=0)
        people_boxes[:, 0] = people_boxes[:, 0] * self.config.img_size[0]
        people_boxes[:, 1] = people_boxes[:, 1] * self.config.img_size[1]
        for i in range(0, people_boxes.size(0)):
            if torch.sum(people_boxes[i]) != 0:
                people_boxes[i][0] = people_boxes[i][0] + (self.config.img_size[0]) * (
                        i // self.config.max_lab % int(w ** (1 / 2)))
                people_boxes[i][1] = people_boxes[i][1] + (self.config.img_size[1]) * (
                        i // (self.config.max_lab * int(w ** (1 / 2))))
        people_boxes[:, 0] = people_boxes[:, 0]
        people_boxes[:, 1] = people_boxes[:, 1]
        people_boxes[:, 2] = (self.config.img_size[0]) * people_boxes[:, 2]
        people_boxes[:, 3] = (self.config.img_size[1]) * people_boxes[:, 3]  # [x,y,w,h]

        # [x,y,w,h] => [x,y,x,y]
        people_boxes[:, 0] = people_boxes[:, 0] - people_boxes[:, 2] / 2
        people_boxes[:, 1] = people_boxes[:, 1] - people_boxes[:, 3] / 2
        people_boxes[:, 2] = people_boxes[:, 0] + people_boxes[:, 2]
        people_boxes[:, 3] = people_boxes[:, 1] + people_boxes[:, 3]

        output = model(union_image)["instances"]
        pred_classes = output.pred_classes
        scores = output.scores
        boxes = output.pred_boxes.tensor
        people_scores = scores[pred_classes == 0]  # select people predict score
        boxes = boxes[pred_classes == 0, :]

        # calculate iou
        attack_image_id = image_index.clone()
        # 0 is normal image, 1 is attack image
        attack_image_id[attack_image_id >= self.config.batch_size] = 1
        attack_image_id[attack_image_id != 1] = 0
        # [batch size] => [batch size, max boxes number] => [batch size * max boxes number]
        attack_image_id = attack_image_id.unsqueeze(-1)
        attack_image_id = attack_image_id.expand((-1, self.config.max_lab)).clone()
        attack_image_id = attack_image_id.reshape(np.prod(attack_image_id.size()))
        needed_boxes = people_boxes.clone()
        needed_boxes = torch.sum(needed_boxes, dim=1)
        attack_image_id = attack_image_id[needed_boxes != 0]

        predict_image_id = attack_image_id.clone().type(torch.float)
        predict_image_id = predict_image_id.unsqueeze(-1)
        predict_image_id = predict_image_id.expand((-1, 2)).clone()
        predict_image_id[:, 1] = torch.abs(predict_image_id[:, 1] - 1)
        people_boxes = people_boxes[needed_boxes != 0]

        for i in range(people_boxes.size(0)):
            if len(boxes) == 0:
                break
            ixmin = torch.maximum(people_boxes[i][0], boxes[:, 0])
            iymin = torch.maximum(people_boxes[i][1], boxes[:, 1])
            ixmax = torch.minimum(people_boxes[i][2], boxes[:, 2])
            iymax = torch.minimum(people_boxes[i][3], boxes[:, 3])
            iw = torch.maximum(ixmax - ixmin + 1., torch.tensor(0., device='cuda'))
            ih = torch.maximum(iymax - iymin + 1., torch.tensor(0., device='cuda'))
            inters = iw * ih
            # union
            uni = ((people_boxes[i][2] - people_boxes[i][0] + 1.) * (people_boxes[i][3] - people_boxes[i][1] + 1.) +
                   (boxes[:, 2] - boxes[:, 0] + 1.) *
                   (boxes[:, 3] - boxes[:, 1] + 1.) - inters)
            overlaps = torch.max(inters / uni)
            overlaps = overlaps.unsqueeze(0)
            overlaps = overlaps.expand((1, 2)).clone()
            overlaps[:, 1] = 1 - overlaps[:, 1]
            # overlaps[1] = 1 - overlaps[1]
            predict_image_id[i] = overlaps
        attack_image_id = attack_image_id.type(torch.long)
        # use temperature parameter to make cross entropy loss converging more better
        predict_image_id = predict_image_id / 0.1
        return predict_image_id, attack_image_id


class PatchEvaluatorOld(nn.Module):
    def __init__(self, model, data_loader):
        super(PatchEvaluatorOld, self).__init__()
        self.config = patch_configs['base']()
        self.calculator = CalculateAP()
        self.data_loader = None
        self.model = None
        self.predicts = None
        self.ground_truths = None
        self.image_sizes = None
        self.register_dataset(model, data_loader)
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()
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
                    image_batch, clothes_boxes_batch, people_boxes_batch, labels_batch, landmarks_batch,
                    segmentations_batch) in enumerate(
                tqdm(self.data_loader)):
                image_batch = image_batch.cuda()
                adv_patch = adv_patch.cuda()
                people_boxes_batch = people_boxes_batch.cuda()
                adv_batch_t = self.patch_transformer(adv_patch, people_boxes_batch, labels_batch)
                p_img_batch = self.patch_applier(image_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.config.img_size[1], self.config.img_size[0]))
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

                    images_ids = torch.cuda.FloatTensor(boxes.size(0), 1).fill_(idx + id * self.config.batch_size)

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

    def forward(self, adv_patch, threshold=0.5, clean=False):
        if not clean:
            predicts = self.inference_on_dataset(adv_patch)
        else:
            predicts = self.inference_on_dataset_clean()
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

    def inference_on_dataset_clean(self):
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
                image_batch = F.interpolate(image_batch, (self.p_config.img_size[1], self.p_config.img_size[0]))
                images = torch.unbind(image_batch, dim=0)
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
                overlaps[overlaps > 1] = 0
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
    import warnings

    warnings.filterwarnings("ignore")
    configs = patch_configs['base']()
    test_data = DataLoader(
        ListDatasetAnn(configs.deepfooling_txt, range_=[0, 160]),
        num_workers=1,
        batch_size=configs.batch_size
    )
    models = [FasterRCNN_R50_C4, FasterRCNN_R_50_DC5, FasterRCNN_R50_FPN, FasterRCNN_R_101_FPN, FasterRCNN, RetinaNet,
              MaskRcnnX152]
    images = os.listdir('./new_patches')
    re = open('result.txt', 'w')
    for image in images:
        print(image)
        re.write('image: ' + str(image) + '\n')
        path = os.path.join('/home/corona/attack/Fooling-Object-Detection-Network/new_patches', image)
        adv = Image.open(path)
        adv = adv.resize((800, 800))
        adv = torchvision.transforms.functional.pil_to_tensor(adv) / 255.

        for item in models:
            model = item()
            patch_evaluator = PatchEvaluator(model, test_data)
            ap = patch_evaluator(adv, clean=False)
            print(ap)
            re.write(str(ap) + '\n')
            del model, patch_evaluator

        print('=' * 50)
        print('=' * 50)
        re.write('=' * 30)
    re.close()

