"""
parse annotations to get information
"""
import cv2
import os
from PIL import Image


class ParseTools:

    def __init__(self):
        pass

    # parse boxes and labels information from one image's annotation file
    # size: [weight, height]
    def parse_boxes_labels(self, size, ann_path,
                           is_xywh2xyxy=False, is_xyxy2xywh=False, is_xywh2xyxy_keep=False):
        with open(ann_path, 'r') as f:
            boxes = f.readlines()
        boxes_output = []
        labels = []
        for box in boxes:
            label = int(box.split(' ')[0])
            box = box.split(' ')[1:]
            box = [float(item.replace('\n', '')) for item in box]
            if len(box) == 0:
                continue
            labels.append(label)
            if is_xywh2xyxy:
                box = self.xywh2xyxy(size, box)
                boxes_output.append(box)
            elif is_xyxy2xywh:
                box = self.xyxy2xywh(box)
                boxes_output.append(box)
            elif is_xywh2xyxy_keep:
                box = self.xywh2xyxy_keep(box)
                boxes_output.append(box)
            else:
                boxes_output.append(box)
        return boxes_output, labels

    # load image information
    def load_image(self, image_path, mode='cv2'
                   , is_xywh2xyxy=False, is_xyxy2xywh=False, is_xyxy2xywh_keep=False) -> dict:
        """
        :param image_path: the path of the image
        :param mode: you can choose use opencv or PIL to read the image
        :param is_xywh2xyxy: box mode xywh ==> xyxy
        :param is_xyxy2xywh: box mode xyxy ==> xywh
        :param is_xyxy2xywh_keep: box mode xyxy ==> xywh (between 0 and 1)
        :return:
        """
        if mode == 'cv2':
            image = cv2.imread(image_path)
            # image_size: [weight, height]
            image_size = (image.shape[1], image.shape[0])
        else:
            image = Image.open(image_path)
            image_size = (image.size[0], image.size[1])
        image_name = image_path.split('/')[-1]
        ann_path = os.path.join("/".join(image_path.split('/')[:-2]), 'labels', image_name.split('.')[0] + '.txt')

        boxes, labels = self.parse_boxes_labels(image_size, ann_path, is_xywh2xyxy, is_xyxy2xywh,
                                                is_xyxy2xywh_keep)
        return {'image': image, 'image_name': image_name, 'image_size': image_size, 'boxes': boxes, 'labels': labels}

    # size: [weight,height] box: (x,y,w,h)
    # return: (x_min, y_min, x_max, y_max)
    def xywh2xyxy(self, size, box) -> tuple:
        dw = size[0]
        dh = size[1]
        w = box[2] * dw
        h = box[3] * dh
        x_min = box[0] * dw - w / 2
        y_min = box[1] * dh - h / 2
        x_max = x_min + w
        y_max = y_min + h
        return (x_min, y_min, x_max, y_max)

    # size: [weight,height] box: (x,y,w,h)
    # return: (x_min, y_min, x_max, y_max)   [0,1]
    def xywh2xyxy_keep(self, box):
        x_min = box[0] - box[2] / 2
        y_min = box[1] - box[3] / 2
        x_max = box[0] + box[2] / 2
        y_max = box[1] + box[3] / 2
        return (x_min, y_min, x_max, y_max)

    def xyxy2xywh(self, box):
        return None

    # load images txt:  there are absolute path
    def load_images_txt(self, txt_file):
        with open(txt_file, 'r') as f:
            images = f.readlines()
            images = [item.replace('\n', '') for item in images]
        return images


if __name__ == '__main__':
    tools = ParseTools()
    tools.load_image('/home/corona/datasets/yolo_person/images/000000444010.jpg')
