import torch
import time
import logging
import datetime
import torch.nn as nn
import numpy as np
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_context, DatasetEvaluators
from detectron2.utils.logger import log_every_n_seconds
from torch.utils.data import DataLoader
from patch import PatchTransformer, PatchApplier
from patch_config import patch_configs
from load_data import ListDataset
from models import RetinaNet


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

    # Inorder to use detectron2's evaluation we need to register our dataset
    def register_dataset(self, model, data_loader):
        self.data_loader = data_loader
        self.model = model
        self.ground_truths, self.image_sizes = self.get_people_dicts(data_loader)

    # get people dicts from pytorch data loader
    def get_people_dicts(self, data_loader):
        dataset_dicts = []
        image_sizes = []
        for idx, (image_batch, boxes_batch, labels_batch) in enumerate(tqdm(data_loader)):
            # [batch,3,416,416] => tuple: [3,416,416] * batch
            boxes_batch = torch.unbind(boxes_batch, dim=0)
            labels_batch = torch.unbind(labels_batch, dim=0)
            for id, (boxes, labels) in enumerate(zip(boxes_batch, labels_batch)):
                image_sizes.append((image_batch[id].size()[1], image_batch[id].size()[2]))
                objs = []
                for box, label in zip(boxes, labels):
                    if label != -1:
                        obj = [label, box[0], box[1], box[2], box[3]]
                        objs.append(obj)
                dataset_dicts.append(np.array(objs))
        return dataset_dicts, image_sizes

    def forward(self, adv_patch):
        predicts = self.inference_on_dataset(adv_patch)
        ap = self.calculator.ap(self.class_id, predicts, self.ground_truths, self.image_sizes)
        return ap

    # rewrite a function in detectron2
    def inference_on_dataset(self, adv_patch):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.forward` accurately.
        The model will be used in eval mode.
        """

        predicts = []
        with torch.no_grad():
            for id, (images, boxes, labels) in enumerate(tqdm(self.data_loader)):
                images = images.cuda()
                boxes = boxes.cuda()
                labels = labels.cuda()
                # apply adv patch on the image
                adv_batch = self.patch_transformer(adv_patch, boxes, labels)
                images = self.patch_applier(images, adv_batch)
                images = torch.unbind(images, dim=0)
                for image in images:
                    outputs = self.model(image)
                    outputs = outputs['instances']
                    boxes = outputs.pred_boxes
                    scores = outputs.scores
                    classes = outputs.pred_classes

                    # select the class we want
                    boxes = boxes[classes == self.class_id].tensor
                    scores = scores[classes == self.class_id]
                    classes = classes[classes == self.class_id]

                    scores = scores.unsqueeze(-1)
                    classes = classes.unsqueeze(-1)
                    predict = torch.cat([scores, boxes], dim=1)
                    predict = torch.cat([classes, predict], dim=1)
                    predict = predict.detach().cpu().numpy()
                    predicts.append(predict)
        return predicts


class CalculateAP:
    def __init__(self):
        pass

    def ap(self, class_id, predicts, ground_truths, image_sizes, threshold=0.5):
        """
        calculate ap of one class
        class id: the class which you want to calculate ap
        predicts: a list contains numpy arrays  np array format [n,6]
        ground_truths: box and label information read from datasets
        image_sizes: [(width,height),(width,height) ... ] recording the images' width and height
        threshold:  box's iou > threshold means this box is a right box
        """
        precisions = []
        recalls = []
        # calculate predictions and recalls
        for predict, ground_truth, image_size in zip(predicts, ground_truths, image_sizes):
            precision, recall = self.precision_recall(class_id, np.array(predict), np.array(ground_truth), image_size,
                                                      threshold)
            precisions.append(precision)
            recalls.append(recall)
        # print(precisions)
        # print(recalls)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
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
        p = p[p[:, 1] >= 0.7]  # remove the low conf boxes, NMS
        ground_truth = ground_truth[ground_truth[:, 0] == class_id]
        conf_sort_id = np.argsort(p[:, 1])[::-1]
        if len(conf_sort_id) == 0:
            return 0, 0
        p = np.delete(p, 1, axis=1)
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
            p_area = (p[i][3] - p[i][1]) * (p[i][4] - p[i][2])
            gt_areas = (gt_[:, 3] - gt_[:, 1]) * (gt_[:, 4] - gt_[:, 2])
            overlap = np.abs(p[i][3] - gt_[:, 1]) * np.abs(p[i][4] - gt_[:, 2])
            min_area = np.copy(gt_areas)
            min_area[min_area > p_area] = p_area

            overlap[overlap > min_area] = 0
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
        return box


if __name__ == '__main__':
    patch_config = patch_configs['base']()
    # load train datasets
    datasets = ListDataset(patch_config.txt_path)
    data = DataLoader(
        datasets,
        batch_size=patch_config.batch_size,
        num_workers=8,
        shuffle=True,
    )
    model = RetinaNet()
    attack_evaluator = PatchEvaluator(model, data)
    patch = Image.open("/home/corona/attack/Fooling-Object-Detection-Network/patches/100_175_3.jpg")
    adv_patch = torch.cuda.FloatTensor(transforms.PILToTensor()(patch).cuda() / 255)
    adv_patch = adv_patch.cuda()
    attack_evaluator(adv_patch)
    print(attack_evaluator.evaluator._predictions)
