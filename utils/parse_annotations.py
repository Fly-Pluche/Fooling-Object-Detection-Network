"""
parse annotations to get information
"""
import cv2
import os
import numpy as np
from PIL import Image
import json


class ParseTools:

    def __init__(self):
        pass

    # parse boxes and labels information from one image's annotation file
    # size: [width, height]
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
            if len(box) == 0 or box == []:
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
    def load_image(self, image_path, mode='numpy'
                   , is_xywh2xyxy=False, is_xyxy2xywh=False, is_xyxy2xywh_keep=False) -> dict:
        """
        :param image_path: the path of the image
        :param mode: you can choose use opencv or PIL to read the image
        :param is_xywh2xyxy: box mode xywh ==> xyxy
        :param is_xyxy2xywh: box mode xyxy ==> xywh
        :param is_xyxy2xywh_keep: box mode xyxy ==> xywh (between 0 and 1)
        :return:
        """
        image, image_size = self.__read_image(image_path, mode)

        image_name = image_path.split('/')[-1]
        ann_path = os.path.join("/".join(image_path.split('/')[:-2]), 'labels', image_name.split('.')[0] + '.txt')

        boxes, labels = self.parse_boxes_labels(image_size, ann_path, is_xywh2xyxy, is_xyxy2xywh,
                                                is_xyxy2xywh_keep)
        return {'image': image, 'image_name': image_name, 'image_size': image_size, 'boxes': boxes, 'labels': labels}

    def __read_image(self, image_path, mode):
        if mode == 'cv2':
            image = cv2.imread(image_path)
            # image_size: [width, height]
            image_size = (image.shape[1], image.shape[0])
        elif mode == 'numpy':
            image = Image.open(image_path)
            image_size = (image.size[0], image.size[1])
            image = np.asarray(image)
        else:
            image = Image.open(image_path)
            image_size = (image.size[0], image.size[1])
        return image, image_size

    # size: [width,height] box: (x,y,w,h)
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

    # size: [width,height] box: (x,y,w,h)
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

    def parse_anno_file(self, anno_path, image_type='jpg', mode='numpy',
                        need_classes=None):
        """
        parse dataset deepfashion's annotation file
        Args:
            anno_path: the path of the anno file
            image_type: choose the image type of the dataset
            mode: you can this to change the reading mode. (numpy rgb,PIL rgb,cv2 bgr)
            need_classes: a list  You can choose the classes and parse its information.
        category_id: a number which corresponds to the category name.
                    In category_id, 1 represents short sleeve top, 2 represents long sleeve top,
                    3 represents short sleeve outwear, 4 represents long sleeve outwear, 5 represents vest,
                    6 represents sling, 7 represents shorts, 8 represents trousers, 9 represents skirt,
                    10 represents short sleeve dress, 11 represents long sleeve dress,
                    12 represents vest dress and 13 represents sling dress.
        style: a number to distinguish between clothing items from images with the same pair id.
               Clothing items with different style numbers from images with the same pair id have
               different styles such as color, printing, and logo.
               In this way, a clothing item from shop images and a clothing item from user image
               are positive commercial-consumer pair if they have the same style number greater than 0
               and they are from images with the same pair id.(If you are confused with style, please refer to issue#10.)
        bounding_box: [x1,y1,x2,y2]，where x1 and y_1 represent the upper left point coordinate of bounding box, '
                      x_2 and y_2 represent the lower right point coordinate of bounding box.
                      (width=x2-x1;height=y2-y1)
        landmarks: [x1,y1,v1,...,xn,yn,vn],
                    where v represents the visibility: v=2 visible; v=1 occlusion; v=0 not labeled.
                    We have different definitions of landmarks for different categories.
                    The orders of landmark annotations are listed in figure 2.
        segmentation: [[x1,y1,...xn,yn],[ ]], where [x1,y1,xn,yn] represents a polygon and a single clothing item may
                      contain more than one polygon.
        scale: a number, where 1 represents small scale, 2 represents modest scale and 3 represents large scale.
        occlusion: a number, where 1 represents slight occlusion(including no occlusion), 2 represents medium occlusion
                   and 3 represents heavy occlusion.
        zoom_in: a number, where 1 represents no zoom-in, 2 represents medium zoom-in and 3 represents lagre zoom-in.
        viewpoint: a number, where 1 represents no wear, 2 represents frontal viewpoint and 3 represents side or back viewpoint.
        """
        with open(anno_path, 'r') as f:
            anno = json.loads(f.read())
        image_name = anno_path.split('.')[-2].split('/')[-1] + '.' + image_type
        image_path = "/".join(anno_path.split('/')[:-2]) + '/image/' + image_name
        # image = self.__read_image(image_path, mode=mode)
        image = None
        segmentation = []
        bounding_boxes = []
        category_id = []
        style = []
        landmarks = []
        scale = []
        occlusion = []
        zoom_in = []
        viewpoint = []
        for item in anno:
            if item[:4] == 'item':
                if (need_classes is None) or (int(anno[item]['category_id']) in need_classes):
                    # This line code may cause lose some parts of clothes but we only need the class of T-shirt.
                    segmentation.append(anno[item]['segmentation'][0])
                    bounding_boxes.append(anno[item]['bounding_box'])
                    category_id.append(anno[item]['category_id'])
                    style.append(anno[item]['style'])
                    landmarks.append(anno[item]['landmarks'])
                    scale.append(anno[item]['scale'])
                    occlusion.append(anno[item]['occlusion'])
                    zoom_in.append(anno[item]['zoom_in'])
                    viewpoint.append(anno[item]['viewpoint'])
        anno_info = {'image': image, 'image_name': image_name, 'segmentations': segmentation,
                     'bounding_boxes': bounding_boxes,
                     'category_ids': category_id, 'styles': style, 'landmarks': landmarks, 'scales': scale,
                     'occlusions': occlusion, 'zoom_ins': zoom_in, 'viewpoints': viewpoint}
        return anno_info


if __name__ == '__main__':
    tools = ParseTools()
    tools.parse_anno_file('/home/ray/data/deepfashion2/validation/annos/014685.json')
