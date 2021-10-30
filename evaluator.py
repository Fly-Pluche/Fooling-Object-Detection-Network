import torch
import copy
import time
import logging
import datetime
import torch.fft as fft
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
from tools import save_predict_image_torch
from load_data import ListDataset
from asr import ObjectVanishingASR
from utils.frequency_tools import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class MaxProbExtractor(nn.Module):
    """
    get the max score in a.json batch of images
    """

    def __init__(self):
        super(MaxProbExtractor, self).__init__()

    def forward(self, model, batch_image, use_nms=False):
        images = torch.unbind(batch_image, 0)
        max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            if use_nms:
                output = model(image, use_nms)["instances"]
            else:
                output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.obj_conf
            people_scores = scores[pred_classes == model.people_index]  # select people predict score
            if len(people_scores) != 0:
                max_prob = torch.max(people_scores)
                max_prob_t[i] = max_prob
        return max_prob_t


class MaxExtractor(nn.Module):
    """
    get the max score and self information of the max iou in a.json batch of images
    """

    def __init__(self):
        super(MaxExtractor, self).__init__()
        self.config = patch_configs['base']()
        self.tools = ParseTools()

    def forward(self, model, batch_image, people_boxes):
        """
        Args:
            model: the model used to predict
            batch_image: a.json batch of images [batch size, 3, width, height]
            people_boxes: a.json batch of boxes [batch size,lab number,4]
        """
        union_image = torchvision.utils.make_grid(batch_image, padding=0)
        # resize union image
        theta = torch.tensor([
            [1, 0, 0.],
            [0, 1, 0.]
        ], dtype=torch.float).cuda()
        N, C, W, H = union_image.unsqueeze(0).size()
        size = torch.Size((N, C, W // 2, H // 2))
        grid = F.affine_grid(theta.unsqueeze(0), size)
        output = F.grid_sample(union_image.unsqueeze(0), grid)
        union_image = output[0]
        # plt.pytorch_imshow(np.array(functional.to_pil_image(union_image)))
        # plt.show()
        # union_image = functional.resize()
        images = torch.unbind(batch_image, 0)
        people_boxes = self.tools.xywh2xyxy_batch_torch(people_boxes, self.config.img_size)
        # max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        max_iou_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.scores
            boxes = output.pred_boxes.tensor
            people_scores = scores[pred_classes == model.people_index]  # select people predict score
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
                max_iou_t[i] = iou_max

        union_detect = model(union_image)['instances']
        labels = union_detect.pred_classes
        max_prob_t_union = union_detect[labels == 0].scores
        max_prob_t = torch.max(max_prob_t_union)
        max_iou_t = torch.mean(max_iou_t)
        return max_prob_t, max_iou_t


class DetMaxExtractor(nn.Module):
    """
    get the max score
    """

    def __init__(self):
        super(DetMaxExtractor, self).__init__()
        self.config = patch_configs['base']()
        self.tools = ParseTools()

    def forward(self, model, batch_image):
        """
        Args:
            model: the model used to predict
            batch_image: a.json batch of images [batch size, 3, width, height]
        """
        # make a union image: concat a batch of images
        # [4,3,600,600] => [3,1200,1200]
        union_image = torchvision.utils.make_grid(batch_image, padding=0, nrow=2)
        # resize union image
        theta = torch.tensor([
            [1, 0, 0.],
            [0, 1, 0.]
        ], dtype=torch.float).cuda()
        N, C, W, H = union_image.unsqueeze(0).size()
        size = torch.Size((N, C, W // 2, H // 2))
        grid = F.affine_grid(theta.unsqueeze(0), size)
        output = F.grid_sample(union_image.unsqueeze(0), grid)
        union_image = output.squeeze(0)
        # plt.pytorch_imshow(np.array(functional.to_pil_image(union_image)))
        # plt.show()
        # union_image = functional.resize()

        # unbind images
        images = torch.unbind(batch_image, 0)

        # init max conf loss
        max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.obj_conf
            people_scores = scores[pred_classes == 0]  # select people predict score
            if len(people_scores) != 0:
                max_prob = torch.max(people_scores)
                max_prob_t[i] = max_prob
        union_detect = model(union_image)['instances']
        labels = union_detect.pred_classes
        max_prob_t_union = union_detect[labels == model.people_index].obj_conf

        # calculate two parts of conf loss
        conf_loss_single_image = torch.mean(max_prob_t)
        conf_loss_union_image = torch.max(max_prob_t_union)
        return conf_loss_single_image, conf_loss_union_image


class ConfMaxExtractor(nn.Module):
    """
    get the max score
    """

    def __init__(self):
        super(ConfMaxExtractor, self).__init__()
        self.config = patch_configs['base']()
        self.tools = ParseTools()

    def forward(self, model, batch_image):
        """
        Args:
            model: the model used to predict
            batch_image: a.json batch of images [batch size, 3, width, height]
        """
        # make a union image: concat a batch of images
        # [4,3,600,600] => [3,1200,1200]
        union_image = torchvision.utils.make_grid(batch_image, padding=0, nrow=2)
        # resize union image
        theta = torch.tensor([
            [1, 0, 0.],
            [0, 1, 0.]
        ], dtype=torch.float).cuda()
        N, C, W, H = union_image.unsqueeze(0).size()
        size = torch.Size((N, C, W // 2, H // 2))
        grid = F.affine_grid(theta.unsqueeze(0), size)
        output = F.grid_sample(union_image.unsqueeze(0), grid)
        union_image = output.squeeze(0)
        # plt.pytorch_imshow(np.array(functional.to_pil_image(union_image)))
        # plt.show()
        # union_image = functional.resize()

        # unbind images
        images = torch.unbind(batch_image, 0)

        # init max conf loss
        max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.scores
            people_scores = scores[pred_classes == 0]  # select people predict score
            if len(people_scores) != 0:
                max_prob = torch.max(people_scores)
                max_prob_t[i] = max_prob
        union_detect = model(union_image)['instances']
        labels = union_detect.pred_classes
        max_prob_t_union = union_detect[labels == model.people_index].scores

        # calculate two parts of conf loss
        conf_loss_single_image = torch.mean(max_prob_t)
        conf_loss_union_image = torch.max(max_prob_t_union)
        return conf_loss_single_image, conf_loss_union_image


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a.json patch.

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

        tvcomp3= torch.sum(torch.abs(adv_patch[:, 1:, 1:] - adv_patch[:, :-1, :-1] + 0.000001), 0)
        tvcomp3 = torch.sum(torch.sum(tvcomp3, 0), 0)
        tv=tv+tvcomp3

        tvcomp4= torch.sum(torch.abs(adv_patch[:, 1:, :-1] - adv_patch[:, :-1, 1:] + 0.000001), 0)
        tvcomp4 = torch.sum(torch.sum(tvcomp4, 0), 0)
        tv=tv+tvcomp4

        return tv / torch.numel(adv_patch)


class FrequencyLoss(nn.Module):
    """TotalVariation: calculates the total variation of a.json patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self, config):
        super(FrequencyLoss, self).__init__()
        self.config = config
        img_h = self.config.patch_size
        img_w = self.config.patch_size
        lpf = torch.zeros((img_h, img_h))
        R = (img_h + img_w) // self.config.fft_size
        for x in range(img_w):
            for y in range(img_h):
                if ((x - (img_w - 1) / 2) ** 2 + (y - (img_h - 1) / 2) ** 2) < (R ** 2):
                    lpf[y, x] = 1
        hpf = 1 - lpf
        self.lpf = lpf.cuda()
        self.hpf = hpf.cuda()

    def forward(self, adv_patch):
        img_low_frequency, img_high_frequency = pytorch_fft(adv_patch, self.lpf, self.hpf)
        loss = torch.sum(img_high_frequency)
        print('img_low_frequency*0.0005',torch.sum(img_low_frequency)*0.0005)

        # loss=loss-torch.sum(img_low_frequency)*0.0008
        # 改权重
        return loss


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val

# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)

class MS_SSIMLOSS(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3, weights=None):
        super(MS_SSIMLOSS, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X):
        X=X.unsqueeze(0)
        Y=X[:,:,1:,1:]
        X=X[:,:,:-1,:-1]
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)

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
        # stitching a.json square picture
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
        # plt.pytorch_imshow(np.array(functional.to_pil_image(union_image)))
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
        people_scores = scores[pred_classes == model.people_index]  # select people predict score
        boxes = boxes[pred_classes == model.people_index, :]

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


class UnionDetectorBCE(nn.Module):
    def __init__(self):
        super(UnionDetectorBCE, self).__init__()
        self.config = patch_configs['base']()

    def forward(self, model, image_batch, p_image_batch, people_boxes):
        """
        Args:
            image_batch: [batch size,3,w,h]
            p_image_batch: [batch size,3,w,h]
            people_boxes: []
        """
        # stitching a.json square picture
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
        # plt.pytorch_imshow(np.array(functional.to_pil_image(union_image)))
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
        people_scores = scores[pred_classes == model.people_index]  # select people predict score
        boxes = boxes[pred_classes == model.people_index, :]

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

        # init predicted image id
        predict_image_id = torch.cuda.FloatTensor(attack_image_id.size()).fill_(1)
        predict_image_id = torch.abs(predict_image_id - 1)
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
            overlaps = 1 - torch.max(inters / uni)
            # overlaps[1] = 1 - overlaps[1]
            predict_image_id[i] = overlaps
        attack_image_id = attack_image_id.type(torch.float)
        # use temperature parameter to make cross entropy loss converging more better
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
                    # a.json = model.visual_instance_predictions(image, outputs)
                    # plt.pytorch_imshow(a.json)
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
    def __init__(self, model, data_loader, use_deformation=True):
        super(PatchEvaluator, self).__init__()
        self.p_config = patch_configs['base']()
        self.calculator = CalculateAP()
        self.data_loader = None
        self.model = None
        self.predicts = None
        self.ground_truths = None
        self.image_sizes = None
        self.use_deformation = use_deformation
        if use_deformation:
            self.patch_transformer = PatchTransformerPro().cuda()
            self.patch_applier = PatchApplierPro().cuda()
        else:
            self.patch_transformer = PatchTransformer().cuda()
            self.patch_applier = PatchApplier().cuda()
        self.class_id = 0  # the class you want to calculate ap
        self.register_dataset(model, data_loader)

    def register_dataset(self, model, data_loader):
        self.data_loader = data_loader
        self.model = model
        self.class_id = self.model.people_index
        self.ground_truths = self.get_people_dicts(data_loader)

    # get people dicts from pytorch data loader
    def get_people_dicts(self, data_loader):
        dataset_dicts = []
        length = len(data_loader)
        data_loader = iter(data_loader)
        for i in tqdm(range(length), ascii=True, desc='load people\'s boxes information'):
            if self.use_deformation:
                image_batch, _, people_boxes, labels_batch, _, _ = next(data_loader)
            else:
                image_batch, people_boxes, labels_batch = next(data_loader)

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
        adv_patch = adv_patch.cuda()
        predicts = self.inference_on_dataset(adv_patch, clean)
        ground_truths = copy.deepcopy(self.ground_truths)
        ap = self.calculator.ap(self.class_id, predicts, ground_truths, self.image_sizes, threshold=threshold)
        return ap

    def save_visual_images(self, adv_patch, root_path, epoch, clean=False):
        file_name = f"{epoch}-"
        index = random.randint(0, 10)
        with torch.no_grad():
            for id, item in enumerate(tqdm(self.data_loader)):
                if id != index:
                    continue
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
                        adv_batch_t = self.patch_transformer(adv_patch, people_boxes, labels_batch)
                        p_img_batch = self.patch_applier(image_batch, adv_batch_t)
                else:
                    p_img_batch = image_batch
                p_img_batch = F.interpolate(p_img_batch, (self.p_config.img_size[0], self.p_config.img_size[1]),
                                            mode='bilinear')
                images = torch.unbind(p_img_batch, dim=0)
                for idx, image in enumerate(images):
                    path = os.path.join(root_path, file_name + str(idx) + '.png')
                    save_predict_image_torch(self.model, image, path)
                if id == index:
                    break

    def inference_on_dataset(self, adv_patch, clean):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.forward` accurately.
        The model will be used in eval mode.
        """
        images_ids_ = []
        confidences_ = []
        BB = []

        with torch.no_grad():
            for id, item in enumerate(tqdm(self.data_loader)):
                if self.use_deformation:
                    image_batch, clothes_boxes_batch, _, _, landmarks_batch, segmentations_batch = item
                    clothes_boxes_batch = clothes_boxes_batch.cuda()
                    landmarks_batch = landmarks_batch.cuda()
                    segmentations_batch = segmentations_batch.cuda()
                else:
                    image_batch, people_boxes, labels_batch = item
                    people_boxes = people_boxes.cuda()
                    labels_batch = labels_batch.cuda()
                image_batch = image_batch.cuda()

                if not clean:
                    if self.use_deformation:
                        adv_batch_t, adv_batch_mask_t = self.patch_transformer(adv_patch, clothes_boxes_batch,
                                                                               segmentations_batch, landmarks_batch,
                                                                               image_batch)
                        p_img_batch = self.patch_applier(image_batch, adv_batch_t, adv_batch_mask_t)
                    else:
                        adv_batch_t = self.patch_transformer(adv_patch, people_boxes, labels_batch)
                        p_img_batch = self.patch_applier(image_batch, adv_batch_t)
                else:
                    p_img_batch = image_batch
                p_img_batch = F.interpolate(p_img_batch, (self.p_config.img_size[1], self.p_config.img_size[0]))
                images = torch.unbind(p_img_batch, dim=0)
                for idx, image in enumerate(images):
                    outputs = self.model(image, nms=True)
                    # a.json = model.visual_instance_predictions(image, outputs)
                    # plt.pytorch_imshow(a.json)
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
                image_batch = F.interpolate(image_batch, (self.p_config.img_size[1], self.p_config.img_size[0]),
                                            mode='bilinear')
                images = torch.unbind(image_batch, dim=0)
                for idx, image in enumerate(images):
                    outputs = self.model(image)
                    # a.json = model.visual_instance_predictions(image, outputs)
                    # plt.pytorch_imshow(a.json)
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

        # avoid divide by zero in case the first detection matches a.json difficult
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
        predict: a.json np array [n, 6]
                 n_: the number of each image's boxes/labels
                 the dim 2 include:
                    label: the id of this class
                    confidence: the confidences of each box
                    box: [x,y,w,h] between 0 and 1
        ground truth: a.json np array with the same size of the predict
        image_sizes: (width,height)  recording the image's width and height
        threshold: box's iou > threshold means this box is a.json right box
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


def calculate_asr(adv, batch_size=2, use_config=True):
    # adv = functional.pil_to_tensor(adv) / 255.0
    config = patch_configs['base']()
    asr_calculate = ObjectVanishingASR(config.img_size, use_deformation=False)
    if use_config:
        batch_size = config.batch_size
    test_data = DataLoader(
        ListDataset(config.coco_val_txt, number=2000),
        num_workers=16,
        batch_size=batch_size
    )
    model = Yolov3(config.model_path, config.model_image_size, config.classes_path)
    model.set_image_size(config.img_size[0])
    asr_calculate.register_dataset(model, test_data)
    asr_calculate.inference_on_dataset(adv, clean=False)
    ASR, ASRs, ASRm, ASRl = asr_calculate.calculate_asr()
    return ASR, ASRs, ASRm, ASRl


if __name__ == '__main__':
    import warnings
    import cv2

    warnings.filterwarnings("ignore")

    img_path = './logs/20210830-151340_base_YOLO_with_coco_datasets/visual_image/260-1.jpg'
    # img_path = './logs/20210909-073432_base_YOLO_with_coco_datasets_use_nms/79.6_asr.jpg'
    adv = Image.open(img_path)
    # adv = np.asarray(adv)
    adv = functional.pil_to_tensor(adv) / 255.0
    adv = adv.cuda()
    h, w = adv.size(1), adv.size(2)
    lpf, hpf = produce_cycle_mask(h, w, 20)
    lpf = lpf.cuda()
    hpf = hpf.cuda()
    plt.imshow(np.asarray(functional.to_pil_image(hpf)))
    plt.show()
    adv_l, adv_h = pytorch_fft(adv, lpf, hpf)
    plt.imshow(np.asarray(functional.to_pil_image(adv_l[0])))
    plt.show()
    plt.imshow(np.asarray(functional.to_pil_image(adv_h[0])))
    plt.show()
    adv = adv_h[0]
    # result = calculate_asr(adv, 2, use_config=False)
    # print(result)

    # config = patch_configs['base']()
    # test_data = DataLoader(
    #     ListDataset(config.coco_val_txt),
    #     num_workers=4,
    #     batch_size=config.batch_size,
    #     shuffle=False
    # )
    # model = Yolov3(model_path='net/yolov3/logs/Epoch10-Total_Loss11.4261-Val_Loss10.1546.pth', image_size=608)
    # # model = Yolov3(model_path='model_data/yolo_weights.pth', image_size=416, classes_path='model_data/coco_classes.txt')
    # model.set_image_size(config.img_size[0])
    # # model.save
    # patch_evaluator = PatchEvaluator(model, test_data, use_deformation=False)
    # from PIL import Image
    #
    # adv = Image.open('./patches/1.jpg')
    # adv = functional.pil_to_tensor(adv) / 255.0
    # adv = adv.cuda()
    # patch_evaluator.save_visual_images(adv, './images/yolo_output', 1, clean=False)
    #
    # ap = patch_evaluator(adv, clean=False)
    # print(ap)

    # model.set_image_size(config.img_size[0])

    # models = [FasterRCNN_R50_C4, FasterRCNN_R_50_DC5, FasterRCNN_R50_FPN, FasterRCNN_R_101_FPN, FasterRCNN, RetinaNet,
    #           MaskRCNN]
    # images = ['R50_DC5.jpg', 'R50_FPN.jpg', 'R101_FPN.jpg', 'R50_C4.jpg']
    # for image in images:
    #     print(image)
    #     path = os.path.join('/home/corona/attack/Fooling-Object-Detection-Network/patches', image)
    #     adv = Image.open(path)
    #     adv = torchvision.transforms.functional.pil_to_tensor(adv) / 255.
    #     for item in models:
    #         model = item()
    #         patch_evaluator = PatchEvaluator(model, test_data)
    #         ap = patch_evaluator(adv, clean=False)
    #         print(ap)
    #         del model, patch_evaluator
    #     print('=' * 50)
    #     print('=' * 50)
    # import cv2
    #
    # # model = FasterRCNN()
    # img = 'images/aaa.jpg'
    # img = cv2.imread(img)
    # # a = model.default_predictor_(img)['instances']
    # # box = a.pred_boxes.tensor
    # # print(box.size())
    # # print(box)
    # img = functional.to_tensor(img)
    # # print(img)
    # model = Yolov3(model_path='net/yolov3/logs/Epoch10-Total_Loss11.4261-Val_Loss10.1546.pth', image_size=608)
    # model.set_image_size(717)
    # result = model(img.cuda(), nms=True)
    # print(result)
    #                  ASR, ASRs, ASRm, ASRl
    # origin image: (0.9102340202584701, 0.7530864197530864, 0.5286624203821656, 0.6960352422907489)
    # only low frequency: (0.9095354523227384, 0.7530864197530864, 0.5316455696202531, 0.691304347826087)
    # cycle size:32: (0.9091861683548725, 0.7530120481927711, 0.5128205128205128, 0.7112068965517241)
    # cycle size: 16:
