from __future__ import absolute_import

import os

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.structures import Instances, Boxes
from torch import nn
from torchvision.transforms import functional

from predict import Predictor
from utils.utils import boxes_scale
from utils.visualizer import Visualizer_
from net.yolov3 import yolo
from net.yolov3.yolo import YoloBody
from net.yolov3.utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                                    yolo_correct_boxes)


# import mmcv


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class BaseModel(nn.Module):
    def __init__(self, model, load_model=True):
        super(BaseModel, self).__init__()
        self.output = None
        if model is not None:
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(model))
            if load_model:
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
                self.predictor = Predictor(self.cfg)
                self.default_predictor_ = DefaultPredictor(self.cfg)
                self.model = self.predictor.model
        else:
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

    def forward(self, img):
        """
        predict instances in the image
        img: torch float [n, C, H, W]
        """
        return self.predictor(img)

    def default_predictor(self, img):
        """
        use detectron2 default yolo_predictor to predict
        img: a.json image read by cv2 or PIL
        """
        return self.default_predictor_(img)

    def visual_instance_predictions(self, img, output, mode='tensor', threshold=0.5):
        """
        draw instance boxes on the image
        """
        v = Visualizer_(img, self, threshold=threshold, mode=mode)
        output = output["instances"].to('cpu')
        out = v.draw_instance_predictions(output)
        return out.get_image()

    def trainer(self, train_datasets_name, num_workers=2, img_per_batch=2, base_lr=1e-3, max_iter=300,
                batch_size_per_image=128, num_classes=1, output_dir='outs'
                ):
        """
        set base information for the trainer
        :param train_datasets_name: the datasets' name you have registered
        :param num_workers: the cpu number you want to choose
        :param img_per_batch:
        :param base_lr: pick a.json good learning rate
        :param max_iter: max train iterations
        :param batch_size_per_image: batch size of the train data
        :param num_classes: total number classes in your datasets
        :return: a.json detectron2 DefaultTrainer. you can use trainer.train() to train the model
        """
        self.cfg.DATASETS.TRAIN = (train_datasets_name,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = img_per_batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = []
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        return trainer


# mask rcnn r50 fpn
class MaskRCNN(BaseModel):
    def __init__(self):
        model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        super(MaskRCNN, self).__init__(model)
        self.model_name = 'MaskRCNN'


class MaskRCNN_PRO(BaseModel):
    def __init__(self):
        model = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
        super(MaskRCNN_PRO, self).__init__(model)
        self.model_name = 'MaskRCNN_PRO'


# faster rcnn x101-FPN
class FasterRCNN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        super(FasterRCNN, self).__init__(model)
        self.model_name = 'FasterRCNN'


# retina net r101
class RetinaNet(BaseModel):
    def __init__(self):
        model = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        super(RetinaNet, self).__init__(model)
        self.model_name = 'RetinaNet'


# fast rcnn
class FastRCNN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
        super(FastRCNN, self).__init__(model)
        self.model_name = 'FastRCNN'


# faster rcnn on pascal voc
class FasterRCNNVOC(BaseModel):
    def __init__(self):
        model = "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"
        super(FasterRCNNVOC, self).__init__(model)
        self.model_name = 'FasterRCNNVOC'


class FasterRCNN_R50_C4(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
        super(FasterRCNN_R50_C4, self).__init__(model)
        self.model_name = 'FasterRCNN_R50_C4'


class FasterRCNN_R_50_DC5(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml"
        super(FasterRCNN_R_50_DC5, self).__init__(model)
        self.model_name = 'FasterRCNN_R_50_DC5'


class FasterRCNN_R50_FPN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        super(FasterRCNN_R50_FPN, self).__init__(model)
        self.model_name = 'FasterRCNN_R50_FPN'


class FasterRCNN_R_101_FPN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        super(FasterRCNN_R_101_FPN, self).__init__(model)
        self.model_name = 'FasterRCNN_R_101_FPN'


# faster rcnn on pascal voc
class MaskRcnnX152(BaseModel):
    def __init__(self):
        model = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
        super(MaskRcnnX152, self).__init__(model)
        self.model_name = 'MaskRcnnX152'


class Yolov3(BaseModel):
    _defaults = {
        # "model_path": 'model_data/yolo_weights.pth',
        "model_path": './net/yolov3/logs/Epoch9-Total_Loss11.5882-Val_Loss10.4279.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "model_image_size": (608, 608, 3),
        "confidence": 0.3,
        "iou": 0.3,
        "cuda": True,
        "letterbox_image": False,
        "confidence_predict": 0.5,
    }

    def __init__(self, model_path=None, image_size=None, classes_path=None):

        self.__dict__.update(self._defaults)

        self.model = None
        if model_path is not None:
            self.model_path = model_path
        if image_size is not None:
            self.model_image_size = (image_size, image_size, 3)
        if classes_path is not None:
            self.classes_path = classes_path
        if 'voc' in self.classes_path:
            self.people_index = 14
            super(Yolov3, self).__init__('PascalVOC-Detection/faster_rcnn_R_50_C4.yaml', False)
        elif 'coco' in self.classes_path:
            self.people_index = 0
            super(Yolov3, self).__init__('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', False)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()
        self.model_name = 'Yolov3'

    def update_model(self, model_path):
        self.model_path = model_path
        self.generate()

    # get all coco classes
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # get all anchor boxes
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    def generate(self):
        self.num_classes = len(self.class_names)
        # create yolo model
        self.model = YoloBody(self.anchors, self.num_classes)

        # load yolo weights
        print(f'Loading weights from {self.model_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval()

        # self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

        # create there feature decoder
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], self.num_classes, (self.model_image_size[1], self.model_image_size[0])))

    def set_image_size(self, img_size):
        self.img_size = img_size

    def yolo_predictor(self, image, nms=False):
        """
        use yolo model to predict
        input: torch tensor [3,w,h], Pixel value range [0,1]
        """
        # resize the input image
        input_img = F.interpolate(image.unsqueeze(0), size=self.model_image_size[0], mode='bilinear')
        # input_img = input_img.type(torch.cuda.FloatTensor)
        # [[x1, y1, x2, y2, confidence, class]]
        outputs = self.model(input_img)
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        # stack the output
        output = torch.cat(output_list, 1)
        if nms:
            with torch.no_grad():
                batch_detections = non_max_suppression(output, self.num_classes, conf_thres=self.confidence_predict,
                                                       nms_thres=self.iou)
                try:
                    output = batch_detections[0]
                except:
                    output = None
        else:
            box_corner = output.new(output.shape)
            box_corner[:, :, 0] = output[:, :, 0] - output[:, :, 2] / 2
            box_corner[:, :, 1] = output[:, :, 1] - output[:, :, 3] / 2
            box_corner[:, :, 2] = output[:, :, 0] + output[:, :, 2] / 2
            box_corner[:, :, 3] = output[:, :, 1] + output[:, :, 3] / 2
            output[:, :, :4] = box_corner[:, :, :4]
            for image_i, image_pred in enumerate(output):
                # ----------------------------------------------------------#
                #   class_conf  [num_anchors, 1]
                #   class_pred  [num_anchors, 1]
                # ----------------------------------------------------------#
                class_conf, class_pred = torch.max(image_pred[:, 5:5 + self.num_classes], 1, keepdim=True)

                # ----------------------------------------------------------#
                #   First-round selection using confidence level
                # ----------------------------------------------------------#
                conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= self.confidence).squeeze()

                image_pred = image_pred[conf_mask]
                class_conf = class_conf[conf_mask]
                class_pred = class_pred[conf_mask]
                if not image_pred.size(0):
                    output = None
                    continue
                # -------------------------------------------------------------------------#
                #   detections  [num_anchors, x1, y1, x2, y2, obj_conf, class_conf, class_pred]
                # -------------------------------------------------------------------------#
                detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
                output = detections

        result = Instances((self.img_size, self.img_size))
        if output is not None:
            pred_boxes = boxes_scale(output[:, 0:4], (self.model_image_size[1], self.model_image_size[0]),
                                     (self.img_size, self.img_size))
            pred_boxes[pred_boxes < 0] = 0
            pred_boxes[pred_boxes > self.img_size] = self.img_size
            boxes = Boxes(pred_boxes)
            obj_conf = output[:, 4]
            class_conf = output[:, 5]
            pred_classes = output[:, 6]
        else:
            pred_boxes = torch.tensor([], device='cuda')
            boxes = Boxes(pred_boxes)
            obj_conf = torch.tensor([], device='cuda')
            class_conf = torch.tensor([], device='cuda')
            pred_classes = torch.tensor([], device='cuda')
        # pred_classes = pred_classes.type(torch.int)

        result.set("pred_boxes", boxes.clone())
        result.set("pred_classes", pred_classes.clone().int())
        result.set("scores", class_conf.clone())
        result.set("obj_conf", obj_conf.clone())
        return {"instances": result}

    def forward(self, image, nms=False):
        return self.yolo_predictor(image, nms)

    @torch.no_grad()
    def default_predictor(self, img):
        """
        directly predict cv2/PIL image
        img: a.json image read by cv2 or PIL
        """
        if type(img) == PIL.JpegImagePlugin.JpegImageFile:
            img = functional.pil_to_tensor(img) / 255.0
        else:
            img = functional.to_tensor(img) / 255.0
        return self.yolo_predictor(img.cuda())


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    img = Image.open('./paper_images/big_high_pig.png')
    # img = img.resize((416, 416))
    yolov3 = Yolov3()
    yolov3.set_image_size(())
    img = functional.pil_to_tensor(img) / 255.0
    img = img.cuda()
    # img = img.unsqueeze(0)
    output = yolov3.yolo_predictor(img)
    print(output)
    img = yolov3.visual_instance_predictions(img[0], output, mode='tensor')
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()
    # output = output['instances'].to('cpu')
    # t = ImageTools()
    # FasterRCNN_R50_C4()
    # FasterRCNN_R_50_DC5()
    # FasterRCNN_R50_FPN()
    # FasterRCNN_R_101_FPN()
    # faster = FasterRCNN()
    # a.json = RetinaNet()
    # b = FastRCNN()
    # c = MaskRCNN()
    # d = FasterRCNNVOC()
    # im = Image.open('/home/corona/datasets/flir_yolo/train/images/06478.jpeg')
    # im = functional.pil_to_tensor(im) / 255.
    #
    # outputs = faster(im)
    # print(outputs)
    # print(type(outputs))
    # print(time.time() - time1)
    # classes = outputs.pred_classes
    # scores = outputs.scores
    # print(classes)
    # print(scores[classes == 0])
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(faster.cfg.DATASETS.TRAIN[0]), scale=1)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # out = out.get_image()[:, :, ::-1]
    # t.cv2_imshow(out, transform=False)
