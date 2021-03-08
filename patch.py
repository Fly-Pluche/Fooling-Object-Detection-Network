from __future__ import absolute_import

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.modules.utils import _pair, _quadruple
from torchvision.transforms import functional

from load_data import ListDatasetAnn
from patch_config import patch_configs
from utils.delaunay2D import Delaunay2D


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class PatchTransformer(nn.Module):
    """
    change patch's contrast and brightness
    rotate the patch
    return a batch patch mask which have the same size with the training image.
    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8  # min contrast
        self.max_contrast = 1.2  # max contrast
        self.min_brightness = -0.1  # min brightness
        self.max_brightness = 0.1  # max brightness
        self.min_angle = -20 / 180 * math.pi  # min angle
        self.max_angle = 20 / 180 * math.pi  # max angle
        self.noise_factor = 0.10  # a limit to noise
        self.configs = patch_configs['base']()
        self.median_pooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, boxes_batch, lab_batch, people_id=0):
        # make people id be 1 other is 0
        for i in range(lab_batch.size()[0]):
            for j in range(lab_batch.size()[1]):
                if lab_batch[i, j] == people_id:
                    lab_batch[i, j] = 1
                else:
                    lab_batch[i, j] = 0
        # make a batch of adversarial patch
        adv_patch = self.median_pooler(adv_patch.unsqueeze(0))
        # determine size of padding
        img_size = np.array(self.configs.img_size)
        # adv_patch is a square
        pad = list((img_size - adv_patch.size(-1)) / 2)
        # a image needs boxes number patch
        adv_batch = adv_patch.unsqueeze(0)
        # a batch adv_batch: (batch size, boxes number, patch.size(0),patch.size(1),patch.size(2))
        adv_batch = adv_batch.expand(boxes_batch.size(0), boxes_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((boxes_batch.size(0), boxes_batch.size(1)))

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # apply contrast, brightness and clamp
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.999999)

        # pad patch and mask to image dimensions
        my_pad = nn.ConstantPad2d((int(pad[0] + 0.5), int(pad[0]), int(pad[1] + 0.5), int(pad[1])), 0)
        adv_batch = my_pad(adv_batch)

        # rotation and rescaling transforms
        angle_size = (lab_batch.size(0) * lab_batch.size(1))
        angle = torch.cuda.FloatTensor(angle_size).uniform_(self.min_angle, self.max_angle)

        # Resizes and rotates the patch
        current_patch_size = adv_patch.size(-1)
        boxes_batch_scaled = torch.cuda.FloatTensor(boxes_batch.size()).fill_(0)
        img_size = list(img_size)
        # box [x,y,w,h]
        boxes_batch_scaled[:, :, 0] = boxes_batch[:, :, 0] * img_size[0]
        boxes_batch_scaled[:, :, 1] = boxes_batch[:, :, 1] * img_size[1]
        boxes_batch_scaled[:, :, 2] = boxes_batch[:, :, 2] * img_size[0]
        boxes_batch_scaled[:, :, 3] = boxes_batch[:, :, 3] * img_size[1]
        target_size = torch.sqrt_(
            ((boxes_batch_scaled[:, :, 2].mul(0.3)) ** 2) + ((boxes_batch_scaled[:, :, 3].mul(0.3)) ** 2)
        )
        target_x = boxes_batch[:, :, 0].view(np.prod(batch_size))
        target_y = boxes_batch[:, :, 1].view(np.prod(batch_size))
        target_off_x = boxes_batch[:, :, 2].view(np.prod(batch_size))
        target_off_y = boxes_batch[:, :, 3].view(np.prod(batch_size))

        # random change the patches' position
        off_x = target_off_x * (torch.cuda.FloatTensor(target_off_x.size()).uniform_(-0.3, 0.3))
        target_x = target_x + off_x
        off_y = target_off_y * (torch.cuda.FloatTensor(target_off_y.size()).uniform_(-0.1, 0.1))
        target_y = target_y + off_y
        target_y = target_y - 0.01
        scale = target_size / current_patch_size
        scale = scale.view(angle_size)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # theta = roataion, rescale matrix
        theta = torch.cuda.FloatTensor(angle_size, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=True)
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0, 0.999999)
        return adv_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        # adv_batch [4, 15, 3, 416, 416]
        for adv in advs:
            img_batch = torch.where((adv > 0.000001), adv, img_batch)
        return img_batch


class PatchVisualor(nn.Module):
    """
    visual batch applying on the origin image
    """

    def __init__(self):
        super(PatchVisualor, self).__init__()
        self.to_cuda = lambda tensor: tensor if tensor.device.type == 'cuda' else tensor.cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()
        self.patch_config = patch_configs['base']()

    def forward(self, image, boxes, labels, adv_patch):
        """
        Args:
            image: a tensor [3,w,h]
            boxes: a tensor [n,4]
            label: a tensor [n]
            adv_patch: a tensor float image [3,w,h] or a path about the patch image
        return: a numpy type image [w, h, 3]
        """
        image = self.to_cuda(image)  # [1,3,w,h]
        boxes = self.to_cuda(boxes)  # [1,n,4]
        labels = self.to_cuda(labels)  # [1,n]
        if isinstance(adv_patch, str):
            adv_patch = Image.open(adv_patch)
            adv_patch = functional.pil_to_tensor(adv_patch)
            adv_patch = functional.resize(adv_patch,
                                          [self.patch_config.patch_size, self.patch_config.patch_size]) / 255.
        adv_patch = self.to_cuda(adv_patch.cuda())
        adv_patch_t = self.patch_transformer(adv_patch, boxes, labels)
        image = self.patch_applier(image, adv_patch_t)  # [1,3,w,h]
        image = image[0].detach().cpu()  # [3,w,h]
        image = np.asarray(functional.to_pil_image(image))  # [w,h,3]
        return image


class PatchGauss(nn.Module):
    """
    Produce patch use 2-dim gauss function
    """

    def __init__(self):
        super(PatchGauss, self).__init__()
        # create x and y coordinates
        self.configs = patch_configs['base']()
        base = torch.cuda.FloatTensor(range(0, self.configs.patch_size))
        x_coordinates = base.unsqueeze(0)
        x_coordinates = x_coordinates.expand(self.configs.patch_size, -1)
        y_coordinates = base.unsqueeze(-1)
        y_coordinates = y_coordinates.expand(-1, self.configs.patch_size)
        self.xy_coordinates = torch.stack([x_coordinates, y_coordinates], dim=0)

    def forward(self, coordinates):
        # the total number of gauss function
        gauss_num = coordinates.size()[1]
        xy_coordinates = self.xy_coordinates.unsqueeze(1)
        # [2,gauss_num,patch_size,patch_size]
        xy_coordinates = xy_coordinates.expand(-1, gauss_num, -1, -1)
        # [2, n]
        coordinates = coordinates * self.configs.patch_size
        # [2,n,1]
        coordinates = coordinates.unsqueeze(-1)
        # [2,n,1,1]
        coordinates = coordinates.unsqueeze(-1)
        # [2,n,patch_size,patch_size]
        adv_patch = coordinates.expand(-1, -1, self.configs.patch_size, self.configs.patch_size).clone()
        # produce the gray background patch
        back_patch = torch.cuda.FloatTensor([0.35])
        back_patch = back_patch.unsqueeze(0)
        back_patch = back_patch.expand(self.configs.patch_size, self.configs.patch_size)
        # calculate gauss function
        adv_patch = adv_patch - xy_coordinates
        adv_patch = adv_patch ** 2
        adv_patch = torch.unbind(adv_patch, dim=0)
        adv_patch = adv_patch[0] + adv_patch[1]
        adv_patch = adv_patch / (-2 * (6.07 ** 2))
        adv_patch = torch.exp(adv_patch) * 1
        adv_patch = self.add(adv_patch)
        adv_patch = adv_patch + back_patch  # add background patch on the origin patch
        # [patch size,patch size] ==> [3, patch size, patch size]
        adv_patch = adv_patch.unsqueeze(0)
        adv_patch = adv_patch.expand(3, -1, -1).clone()
        adv_patch = adv_patch.clamp(0, 1)
        return adv_patch

    def add(self, adv_patch):
        adv_patch = list(torch.unbind(adv_patch, dim=0))
        for i in range(1, len(adv_patch)):
            adv_patch[0] = torch.add(adv_patch[0], adv_patch[i])
        return adv_patch[0]


class PatchDelaunay2D:
    """
    create Delaunay2D triangles for patch.
    """

    def __init__(self):
        self.patch_config = patch_configs['base']()
        self.dt = Delaunay2D()
        self.seeds_num = 0
        self.seeds, self.dt_tris = self.create_patch_delaunay()

        # self.seeds = self.seeds / self.patch_config.patch_size

    def create_patch_delaunay(self):
        """
        use delaunay to create triangles,
        return:
            seeds: the [x,y] of points, type: float 0-1
            tris: triangles' points
        """
        radius = self.patch_config.patch_size
        seeds = []
        num = 4
        self.seeds_num = (num + 1) ** 2
        for i in range(0, num + 1):
            x = i * self.patch_config.patch_size / num
            for j in range(0, num + 1):
                y = j * self.patch_config.patch_size / num
                seeds.append([x, y])
        seeds = np.array(seeds, dtype=np.float)
        center = np.mean(seeds, axis=0)
        self.dt = Delaunay2D(center, 50 * radius)

        # Insert all seeds one by one
        for s in seeds:
            self.dt.addPoint(s)

        dt_tris = np.array(self.dt.exportTriangles())
        print(dt_tris)
        return seeds, dt_tris

    def visual(self):
        # Create a plot with matplotlib.pyplot
        fig, ax = plt.subplots()
        ax.margins(0.1)
        ax.set_aspect('equal')
        plt.axis([-1, self.patch_config.patch_size + 1, -1, self.patch_config.patch_size + 1])
        # Plot our Delaunay triangulation (plot in blue)
        cx, cy = zip(*self.seeds)
        ax.triplot(matplotlib.tri.Triangulation(cx, cy, self.dt_tris), 'bo--')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()

    def write_to_obj(self, file_path):
        vertices_num = len(self.seeds)
        faces_num = len(self.dt_tris)
        seeds = self.seeds / self.patch_config.patch_size
        obj = ''
        obj += '#vertices: %d\n' % vertices_num
        obj += '#faces: %d\n' % faces_num
        for seed in seeds:
            obj += 'v '
            obj += str(seed[0]) + ' '
            obj += str(seed[1])
            obj += '\n'
        for tri in self.dt_tris:
            obj += 'f '
            obj += str(tri[0]) + ' '
            obj += str(tri[1]) + ' '
            obj += str(tri[2])
            obj += '\n'
        with open(file_path, 'w') as f:
            f.write(obj)


class Computer:
    def __init__(self):
        self.batch_size = 2

    @staticmethod
    def findNeighbour(edge, faces):
        # find the four neighbouring vertices of the edge
        neighbours = [np.nan, np.nan]
        count = 0
        for i, face in enumerate(faces):
            if np.any(face == edge[0]):
                if np.any(face == edge[1]):
                    neighbourI = np.where(face[np.where(face != edge[0])] != edge[1])
                    neighbourI = neighbourI[0][0]
                    n = face[np.where(face != edge[0])]
                    neighbours[count] = int(n[neighbourI])
                    count += 1
        lI, rI = neighbours
        return lI, rI

    def computeG(self, vertices, edges, faces):
        # G = np.zeros((np.size(edges,0), 8,4))
        gCalc = np.zeros((np.size(edges, 0), 2, 8))
        GI = np.zeros((np.size(edges, 0), 4))
        for k, edge in enumerate(edges):
            iI = int(edge[0])
            jI = int(edge[1])
            i = vertices[iI - 1, :]
            j = vertices[jI - 1, :]
            lI, rI = self.findNeighbour(edge, faces)
            l = vertices[lI - 1, :]
            if not np.isnan(rI):
                r = vertices[rI - 1, :]

            if np.isnan(rI):
                g = np.array([[i[0], i[1], 1, 0],
                              [i[1], -i[0], 0, 1],
                              [j[0], j[1], 1, 0],
                              [j[1], -j[0], 0, 1],
                              [l[0], l[1], 1, 0],
                              [l[1], -l[0], 0, 1]])
                # G[k,:,:] = g
                GI[k, :] = [iI, jI, lI, np.nan]
                gTemp = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
                gCalc[k, :, 0:6] = gTemp[0:2, :]

            else:
                g = np.array([[i[0], i[1], 1, 0],
                              [i[1], -i[0], 0, 1],
                              [j[0], j[1], 1, 0],
                              [j[1], -j[0], 0, 1],
                              [l[0], l[1], 1, 0],
                              [l[1], -l[0], 0, 1],
                              [r[0], r[1], 1, 0],
                              [r[1], -r[0], 0, 1]])
                # G[k,:,:]
                GI[k, :] = [iI, jI, lI, rI]
                gTemp = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
                gCalc[k, :, :] = gTemp[0:2, :]
        return GI, gCalc

    def computeH(self, edges, gCalc, GI, vertices):
        H = np.zeros((np.size(edges, 0) * 2, 8))
        for k, edge in enumerate(edges):

            ek = vertices[int(edge[1]) - 1, :] - vertices[int(edge[0]) - 1, :]
            EK = [[ek[0], ek[1]], [ek[1], -ek[0]]]

            if np.isnan(GI[k, 3]):
                oz = [[-1, 0, 1, 0, 0, 0],
                      [0, -1, 0, 1, 0, 0]]
                g = gCalc[k, :, 0:6]
                # gCalc = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
                hCalc = oz - (EK @ g)
                H[k * 2, 0:6] = hCalc[0, :]
                H[k * 2 + 1, 0:6] = hCalc[1, :]
            else:
                oz = [[-1, 0, 1, 0, 0, 0, 0, 0],
                      [0, -1, 0, 1, 0, 0, 0, 0]]
                g = gCalc[k, :, :]
                # gCalc = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
                hCalc = oz - (EK @ g)
                H[k * 2, :] = hCalc[0, :]
                H[k * 2 + 1, :] = hCalc[1, :]
        return H

    def computeVPrime(self, edges, vertices, GI, H, C, locations):
        A = np.zeros((np.size(edges, 0) * 2 + np.size(C) * 2, np.size(vertices, 0) * 2))
        b = np.zeros((np.size(edges, 0) * 2 + np.size(C) * 2, 1))

        w = 1000

        vPrime = np.zeros((np.size(vertices, 0), 2))

        for gIndex, g in enumerate(GI):
            for vIndex, v in enumerate(g):
                if not np.isnan(v):
                    v = int(v) - 1
                    A[gIndex * 2, v * 2] = H[gIndex * 2, vIndex * 2]
                    A[gIndex * 2 + 1, v * 2] = H[gIndex * 2 + 1, vIndex * 2]
                    A[gIndex * 2, v * 2 + 1] = H[gIndex * 2, vIndex * 2 + 1]
                    A[gIndex * 2 + 1, v * 2 + 1] = H[gIndex * 2 + 1, vIndex * 2 + 1]

        for cIndex, c in enumerate(C):
            A[np.size(edges, 0) * 2 + cIndex * 2, c * 2] = w
            A[np.size(edges, 0) * 2 + cIndex * 2 + 1, c * 2 + 1] = w
            b[np.size(edges, 0) * 2 + cIndex * 2] = w * locations[cIndex, 0]
            b[np.size(edges, 0) * 2 + cIndex * 2 + 1] = w * locations[cIndex, 1]

        V = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]

        vPrime[:, 0] = V[0::2, 0]
        vPrime[:, 1] = V[1::2, 0]

        return vPrime, A, b

    def computeVPrimeFast(self, edges, vertices, C, locations, A, b):
        w = 1000

        vPrime = np.zeros((np.size(vertices, 0), 2))

        V = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]

        vPrime[:, 0] = V[0::2, 0]
        vPrime[:, 1] = V[1::2, 0]

        return vPrime


class PatchARAPTransformer(nn.Module):
    def __init__(self):
        """
        use ARAP to transformer the image
        """
        super(PatchARAPTransformer, self).__init__()
        self.computer = Computer()
        self.patch_config = patch_configs['base']()
        self.patch_delaunay2d = PatchDelaunay2D()
        self.needed_points = [11, 12, 13, 14, 15, 16, 17, 18, 19]

    def numpy_expand(self, array):
        array = torch.from_numpy(array)
        array = array.unsqueeze(-1)
        array = array.unsqueeze(-1)
        array = array.expand((-1, -1, self.patch_delaunay2d.seeds_num, 2)).numpy()
        return array

    def forward(self, adv_patch, boxes_batch, segmentations_batch, points_batch):
        batch_size = boxes_batch.size()[0]
        boxes_number = boxes_batch.size()[1]

        # transform adv patch one the boxes' positions
        # load base Delaunay coordinates
        seeds_basic = np.copy(self.patch_delaunay2d.seeds)
        tris = np.copy(self.patch_delaunay2d.dt_tris)

        # pad the adv_patch and make it have the size size with image
        # [3,w,h] ==> [1,3,w,h]
        adv_patch = adv_patch.unsqueeze(0)

        # adv_patch = adv_patch.expand(boxes_number, -1, -1, -1)
        # [1,3,w,h] ==> [1,3,w,h]
        adv_patch = adv_patch.unsqueeze(0)
        # [1,1,3,w,h] ==> [batch size,boxes num,3,w,h]
        adv_batch = adv_patch.expand(batch_size, boxes_number, -1, -1, -1)
        img_size = np.array(self.patch_config.img_size_big)
        pad = list((img_size - adv_patch.size(-1)) / 2)
        my_pad = nn.ConstantPad2d((int(pad[0] + 0.5), int(pad[0]), int(pad[1] + 0.5), int(pad[1])), 0)
        adv_batch = my_pad(adv_batch)

        # pad seeds and make its value between 0 and 1
        seeds_basic[:, 0] = seeds_basic[:, 0] + int(pad[0] + 0.5)
        seeds_basic[:, 1] = seeds_basic[:, 1] + int(pad[1] + 0.5)
        seeds_basic[:, 0] /= self.patch_config.img_size_big[0]
        seeds_basic[:, 1] /= self.patch_config.img_size_big[1]

        # according box to calculate the basic size of the delaunay coordinates
        # find needed points:  [12,13,14,15,16,17,18,19,20]
        useful_points = np.copy(points_batch) * self.patch_config.img_size_big[0]  # [batch, 2, 25, 3]
        useful_points = useful_points[:, :, self.needed_points, :]
        xy_center = (useful_points[:, :, 1, :] + useful_points[:, :, 2, :] +
                     useful_points[:, :, 6, :] + useful_points[:, :, 7, :]) / 4
        x_center = xy_center[:, :, 0]  # [4,2]
        y_center = xy_center[:, :, 1]  # [4,2]
        x_center = self.numpy_expand(x_center)  # [4,2,seeds_num,2]
        y_center = self.numpy_expand(y_center)  # [4,2,seeds_num,2]
        w1 = (useful_points[:, :, 7, 0] + useful_points[:, :, 6, 0]) / 2 - (
                useful_points[:, :, 2, 0] + useful_points[:, :, 1, 0]) / 2
        w2 = (useful_points[:, :, 3, 1] - useful_points[:, :, 2, 1]) / 2.7 + useful_points[:, :, 2, 1] \
             - useful_points[:, :, 1, 1] + (useful_points[:, :, 1, 1] - useful_points[:, :, 0, 1]) / 2.7
        # w: [batch, boxes number] int
        w = np.min((w1, w2), axis=0)
        w = self.numpy_expand(w)

        # change delaunay coordinates
        # seeds for each box
        seeds_batch = np.copy(self.patch_delaunay2d.seeds)  # [seeds_num, 2]
        seeds_batch = torch.from_numpy(seeds_batch)
        seeds_batch = seeds_batch.unsqueeze(0)  # [1,seeds_num,2]
        seeds_batch = seeds_batch.unsqueeze(0)  # [1,1,seeds_num,2]
        # [batch,boxes_number,seeds_num,2]
        seeds_batch = seeds_batch.expand((batch_size, boxes_number, -1, -1)).numpy()
        seeds_batch = seeds_batch / self.patch_config.patch_size
        seeds_batch = seeds_batch * w

        # move to the center point of the box
        x_pad = x_center - w / 2
        y_pad = y_center - w / 2
        seeds_batch[:, :, :, 0] = seeds_batch[:, :, :, 0] - w[:, :, :, 0] / 2
        seeds_batch[:, :, :, 1] = seeds_batch[:, :, :, 1] - w[:, :, :, 0] / 2

        # rotate the seeds batch
        #  y15-y13   y17-y19     1
        # (——————— + ———————) * ———
        #  x15-x13   x17-x19     2
        left_angels = np.arctan((useful_points[:, :, 3, 1] - useful_points[:, :, 1, 1]) / (
                useful_points[:, :, 3, 0] - useful_points[:, :, 1, 0]))
        right_angels = np.arctan((useful_points[:, :, 5, 1] - useful_points[:, :, 7, 1]) / (
                useful_points[:, :, 5, 0] - useful_points[:, :, 7, 0]))
        angels = (left_angels + right_angels) / 2
        angels = self.numpy_expand(angels)
        rotated_seeds_batch = np.copy(seeds_batch)

        # x1=cos(angle)*x-sin(angle)*y
        rotated_seeds_batch[:, :, :, 0] = \
            np.cos(angels[:, :, :, 0]) * seeds_batch[:, :, :, 0] - np.sin(angels[:, :, :, 0]) * seeds_batch[:, :, :, 1]
        # y1=cos(angle)*y+sin(angle)*x
        rotated_seeds_batch[:, :, :, 1] = \
            np.cos(angels[:, :, :, 0]) * seeds_batch[:, :, :, 1] + np.sin(angels[:, :, :, 0]) * seeds_batch[:, :, :, 0]
        rotated_seeds_batch[:, :, :, 1] = rotated_seeds_batch[:, :, :, 1] + y_pad[:, :, :, 0] + w[:, :, :, 0] / 2
        rotated_seeds_batch[:, :, :, 0] = rotated_seeds_batch[:, :, :, 0] + x_pad[:, :, :, 0] + w[:, :, :, 0] / 2
        self.seeds_batch = rotated_seeds_batch

        # load tensor on the gpu
        rotated_seeds_batch = torch.tensor(rotated_seeds_batch).cuda()  # triangles' coordinates after transforming
        tris = torch.tensor(tris).cuda()
        seeds_basic = torch.tensor(seeds_basic).cuda()  # triangles' coordinates before transforming



        # TODO: affine transformation: move the image to the right place (move each small triangles)

        # TODO: get clothe shadow according the segmentations of the clothe
        # TODO: apply shadow on the adv_patch
        pass


if __name__ == '__main__':
    # data_loader = load_test_data_loader('/home/corona/datasets/WiderPerson/train/train.txt', 10)
    # images, boxes, labels = list(iter(data_loader))[6]
    # adv_patch = '/home/corona/attack/Fooling-Object-Detection-Network/patches/fg_patch.png'
    # origin_image = np.array(functional.to_pil_image(images[0]))
    # visualor = PatchVisualor()
    # model = RetinaNet()
    # image = visualor(images, boxes, labels, adv_patch)
    # out = model.default_predictor(image)
    # img = model.visual_instance_predictions(image, out, mode='rgb')
    # out2 = model.default_predictor(origin_image)
    # img2 = model.visual_instance_predictions(origin_image, out2, mode='rgb')
    # gray_patch = torch.full((3, 200, 200), 0.5)
    # gray = visualor(images, boxes, labels, gray_patch)
    # out3 = model.default_predictor(gray)
    # img3 = model.visual_instance_predictions(gray, out3, mode='rgb')
    # image2 = visualor(images, boxes, labels, 'images/fg.jpeg')
    # out4 = model.default_predictor(image2)
    # img4 = model.visual_instance_predictions(image2, out4, mode='rgb')
    # plt.imshow(img2)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img3)
    # plt.show()
    # plt.imshow(img4)
    # plt.show()
    # image = PatchDelaunay2D()
    # print(len(image.seeds))
    # image.visual()
    # image.write_to_obj('./box.obj')
    # print(image.dt_tris)
    image_id = 2
    config = patch_configs['base']()
    da = ListDatasetAnn(config.deepfashion_txt, 10)
    loader = torch.utils.data.DataLoader(
        da,
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=True,
    )
    image = Image.open('/home/corona/attack/Fooling-Object-Detection-Network/patches/fg_patch.png')
    adv = functional.pil_to_tensor(image)
    images_batch, boxes_batch, labels_batch, landmarks_batch = next(iter(loader))
    AA = PatchARAPTransformer()
    AA(adv, boxes_batch, 1, landmarks_batch)
    images_batch = images_batch[image_id]
    a = np.array(functional.to_pil_image(images_batch))
    fig, ax = plt.subplots(1)
    ax.imshow(a)
    import matplotlib.patches as patches

    landmarks_batch = landmarks_batch * config.img_size_big[0]
    boxes_batch = boxes_batch * config.img_size_big[0]
    boxes_batch = boxes_batch[image_id][0]
    # boxes_batch[0] = boxes_batch[0] - boxes_batch[2]/2
    # boxes_batch[1] = boxes_batch[1] - boxes_batch[3]/2
    # boxes_batch[2] = boxes_batch[2]/2 + boxes_batch[0]
    # boxes_batch[3] = boxes_batch[3]/2 + boxes_batch[1]
    rect = patches.Rectangle(xy=(boxes_batch[0] - boxes_batch[2] / 2, boxes_batch[1] - boxes_batch[3] / 2),
                             width=boxes_batch[2],
                             height=boxes_batch[3], linewidth=2, fill=False, edgecolor='r')
    ax.add_patch(rect)
    # rect = patches.Rectangle(xy=(325 - 162 / 2, 293 - 162 / 2),
    #                          width=162,
    #                          height=162, linewidth=2, fill=False, edgecolor='r')
    useful_points = landmarks_batch[:, :, AA.needed_points, :][image_id][0]
    x = useful_points[:, 0].numpy()
    y = useful_points[:, 1].numpy()
    print('x:', x)
    print('y:', y)
    plt.plot(x, y, 'o')
    ax.add_patch(rect)
    points = AA.seeds_batch
    points = points[image_id][0]
    cx = points[:, 0]
    cy = points[:, 1]
    # print(cx)
    ax.triplot(matplotlib.tri.Triangulation(cx, cy, AA.patch_delaunay2d.dt_tris), 'bo--', markersize=0.5)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
