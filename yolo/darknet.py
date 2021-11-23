from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import cv2





def get_test_input(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (608, 608))
    # BGR => RGB
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    return img_


def parse_cfg(cfg_file_path):
    """
    获取配置文件

    返回一个block列表。每个block描述神经元中的一个块
    block在列表中表示为字典
    """
    file = open(cfg_file_path, 'r')
    lines = file.read().split('\n')  # 将lines按行拆分储存在list中
    lines = [x for x in lines if len(x) > 0]  # 去除空行
    lines = [x for x in lines if x[0] != '#']  # 去除注释
    lines = [x.rstrip().lstrip() for x in lines]  # 去除多余的空格

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':  # 这标志着一个新block的开始
            if len(block) != 0:  # 如果block不是空的，则意味着它存储了前一个block的值。
                blocks.append(block)  # 添加到 blocks list 里
                block = {}  # 初始化block
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def create_modules(blocks):
    net_info = blocks[0]  # 捕获有关输入和预处理的信息, 也就是net里面的信息
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # 添加BN层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # 激活函数
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)

        # 上采样层
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # 位置标注
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut layer
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)

        # yolo 检测层 detection layer
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Darknet(nn.Module):
    def __init__(self, cfg_file_path):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file_path)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # 我们为路由层缓存输出
        write = 0  # 这个过会解释
        for i, module in enumerate(modules):
            module_type = (module['type'])
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                # layers 长度为1时不需要进行feature map的拼接
                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                # 需要将两个feature maps进行拼接
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # 获取输入纬度
                inp_dim = int(self.net_info['height'])

                # 获取类别数量
                num_classes = int(module['classes'])

                # 转化
                x = x.data

                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x
        return detections

    def load_weights(self, weight_file):
        fp = open(weight_file, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # 如果module_type时convolutional则加载权重
            # 否则就跳过
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_norm = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_norm = 0
                conv = model[0]
                if batch_norm:
                    bn = model[1]

                    # 获取BN层的权重参数数量
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 获取权重参数
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.get_D2)
                    bn_weights = bn_weights.view_as(bn.weight.get_D2)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.get_D2.copy_(bn_biases)
                    bn.weight.get_D2.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.get_D2)

                    # Finally copy the data
                    conv.bias.get_D2.copy_(conv_biases)

                # 为卷积层加载权重
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.get_D2)
                conv.weight.get_D2.copy_(conv_weights)


if __name__ == '__main__':
    blocks = parse_cfg('cfg/yolov3.cfg')
    # print(create_modules(blocks))
    model = Darknet('cfg/yolov3.cfg').cuda()
    model.load_weights('yolov3.weights')
    inp = get_test_input('dog-cycle-car.png').cuda()
    pred = model(inp, torch.cuda.is_available())
    a = write_results(pred, 0.8, 80)
    print(a)
