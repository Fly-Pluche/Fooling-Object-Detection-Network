from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        # route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            end = int(x['layers'][-1])
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
                filters = output_filter[index + start]

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



if __name__ == '__main__':
    blocks = parse_cfg('cfg/yolov3.cfg')
    print(create_modules(blocks))
