import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import cv2

YOLO_ROOT = '/home/disk2/ray/datasets/coco2017/yolo'
TRAIN_TXT = os.path.join(YOLO_ROOT, 'train2017.txt')
VAL_TXT = os.path.join(YOLO_ROOT, 'val2017.txt')


def load_images(images_path):
    for item in images_path:
        image_name = item.split('/')[-1].split('.')[0]
        label_path = os.path.join(YOLO_ROOT, 'labels', image_name + '.txt')
        yield item, label_path


def load_label_info(label_path):
    classes = []
    xs = []
    ys = []
    ws = []
    hs = []
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for label in labels:
            label = label.split(' ')
            classes.append(int(label[0]))
            xs.append(float(label[1]))
            ys.append(float(label[2]))
            ws.append(float(label[3]))
            hs.append(float(label[4]))
        classes = np.array(classes)
        xs = list(np.array(xs)[classes == 0])
        ys = list(np.array(ys)[classes == 0])
        ws = list(np.array(ws)[classes == 0])
        hs = list(np.array(hs)[classes == 0])
        return classes, xs, ys, ws, hs


def visual_boxes_status(path):
    pass


def calculate_threshold(path):
    with open(path, 'r') as f:
        images_path = f.readlines()
    num = 0
    total_area = 0
    for item in tqdm(images_path):
        image_name = item.split('/')[-1].split('.')[0]
        label_path = os.path.join(YOLO_ROOT, 'labels', image_name + '.txt')
        # image_path, label_path = item
        classes, xs, ys, ws, hs = load_label_info(label_path)
        for w_, h_ in zip(ws, hs):
            if 0.5 < h_ < 0.55:
                area = w_ * h_
                total_area += area
                num += 1

    print(total_area / num)


def select(path):
    result = ''
    with open(path, 'r') as f:
        images_path = f.readlines()
    # loader = load_images(images_path)
    # result = {}
    w_ = []
    h_ = []
    classes_ = []
    for item in tqdm(images_path):
        image_name = item.split('/')[-1].split('.')[0]
        label_path = os.path.join(YOLO_ROOT, 'labels', image_name + '.txt')
        # image_path, label_path = item
        classes, xs, ys, ws, hs = load_label_info(label_path)
        if len(xs) > 5:
            continue

        if len(xs) == 0:
            continue
        flag = 0
        for h in hs:
            if h <= 0.30 or h >= 0.70:
                flag = 1
        if flag:
            continue
        # print(item)
        result += item
        w_.extend(ws)
        h_.extend(hs)
        classes_.extend(classes)

    # classes_ = np.array(classes_)
    # w_ = np.array(w_)[classes_ == 0]
    # h_ = np.array(h_)[classes_ == 0]
    # plt.hist(x=w_, bins=100,
    #          color="steelblue",
    #          edgecolor="black")
    # plt.show()
    plt.hist(x=h_, bins=100,
             color="steelblue",
             edgecolor="black")
    plt.show()
    print(len(h_))
    if 'train' in path:
        save_path = '/home/disk2/ray/datasets/coco2017/yolo/train3.txt'
    else:
        save_path = '/home/disk2/ray/datasets/coco2017/yolo/val3.txt'
    with open(save_path, 'w') as f:
        f.write(result)


# def run():


if __name__ == '__main__':
    calculate_threshold('/home/disk2/ray/datasets/coco2017/yolo/train3.txt')
    # 0.08159471524693583
    #
    # select(VAL_TXT)
    # with open('/home/disk2/ray/datasets/coco2017/yolo/train_new.txt', 'r') as f:
    #     result = f.readlines()
    #
    # print(len(result))
    # with open('/home/disk2/ray/datasets/coco2017/yolo/test_new.txt', 'r') as f:
    #     result = f.readlines()
    # print(len(result))
    # from PIL import Image
    #
    # for i in range(10):
    #     a = result[i]
    #     img = Image.open(a.strip())
    #     plt.imshow(np.asarray(img))
    #     plt.show()
    # total = len(result)
    # train_num = int(total * 0.8)
    # train = result[0:train_num]
    # test = result[train_num:]
    # train = "".join(train)
    # test = "".join(test)
    # with open('/home/disk2/ray/datasets/coco2017/yolo/train_new.txt', 'w') as f:
    #     f.write(train)
    # with open('/home/disk2/ray/datasets/coco2017/yolo/test_new.txt', 'w') as f:
    #     f.write(test)
    # print(len(result))
# result = set(result)
# result = list(result)
# print(len(result))
# from PIL import Image
#
# img = Image.open('/home/disk2/ray/datasets/coco2017/yolo/images/000000056550.jpg')
# plt.imshow(np.asarray(img))
# plt.show()
