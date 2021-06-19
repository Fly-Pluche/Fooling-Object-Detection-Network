from __future__ import absolute_import

import os

import PIL
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
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import non_max_suppression
from utils.utils import boxes_scale
from utils.visualizer import Visualizer_


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class BaseModel(nn.Module):
    def __init__(self, model):
        super(BaseModel, self).__init__()
        self.output = None
        if model is not None:
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(model))
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
        use detectron2 default predictor to predict
        img: a image read by cv2 or PIL
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
        :param base_lr: pick a good learning rate
        :param max_iter: max train iterations
        :param batch_size_per_image: batch size of the train data
        :param num_classes: total number classes in your datasets
        :return: a detectron2 DefaultTrainer. you can use trainer.train() to train the model
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


# faster rcnn x101-FPN
class FasterRCNN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        super(FasterRCNN, self).__init__(model)


# retina net r101
class RetinaNet(BaseModel):
    def __init__(self):
        model = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        super(RetinaNet, self).__init__(model)


# fast rcnn
class FastRCNN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
        super(FastRCNN, self).__init__(model)


# faster rcnn on pascal voc
class FasterRCNNVOC(BaseModel):
    def __init__(self):
        model = "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"
        super(FasterRCNNVOC, self).__init__(model)


class FasterRCNN_R50_C4(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
        super(FasterRCNN_R50_C4, self).__init__(model)


class FasterRCNN_R_50_DC5(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml"
        super(FasterRCNN_R_50_DC5, self).__init__(model)


class FasterRCNN_R50_FPN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        super(FasterRCNN_R50_FPN, self).__init__(model)


class FasterRCNN_R_101_FPN(BaseModel):
    def __init__(self):
        model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        super(FasterRCNN_R_101_FPN, self).__init__(model)


# faster rcnn on pascal voc
class MaskRcnnX152(BaseModel):
    def __init__(self):
        model = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
        super(MaskRcnnX152, self).__init__(model)


class Yolov3(BaseModel):
    def __init__(self, model_path='config/yolov3.cfg', weights_path='weights/yolov3.weights'):
        super(Yolov3, self).__init__(None)
        self.model_path = model_path
        self.weights_path = weights_path
        self.model = self.build()
        self.img_size = 416
        self.conf_thres = 0.5
        self.nms_thres = 0.5

    def predictor(self, image):
        """
        use yolov3 model to predict
        input: torch tensor [1,3,w,h], Pixel value range [0,1]
        """
        # resize the input image
        input_img = F.interpolate(image.unsqueeze(0), size=416, mode="nearest")
        input_img = input_img.type(torch.cuda.FloatTensor)
        # [[x1, y1, x2, y2, confidence, class]]
        detections = self.model(input_img)
        # nms
        detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]
        result = Instances((self.img_size, self.img_size))
        pred_boxes = boxes_scale(detections[:, 0:4], (416, 416), (self.img_size, self.img_size))
        boxes = Boxes(pred_boxes)
        scores = detections[:, 4]
        pred_classes = detections[:, 5]
        pred_classes = pred_classes.type(torch.int)

        result.set("pred_boxes", boxes)
        result.set("pred_classes", pred_classes)
        result.set("scores", scores)
        return {"instances": result}

    def forward(self, image):
        # Configure input
        # input_img = F.interpolate(image.unsqueeze(0), size=self.img_size, mode="nearest")
        # input_img = input_img / 255
        # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # input_img = Variable(input_img.type(Tensor))
        # detections = self.model(input_img)
        # detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        # detections = detections[0]
        # print(detections[:, 0:4])
        return self.predictor(image)

    @torch.no_grad()
    def default_predictor(self, img):
        """
        directly predict cv2/PIL image
        img: a image read by cv2 or PIL
        """
        if type(img) == PIL.JpegImagePlugin.JpegImageFile:
            img = functional.pil_to_tensor(img) / 255.0
        else:
            img = functional.to_tensor(img) / 255.0
        return self.predictor(img)

    def build(self):
        model = load_model(self.model_path, self.weights_path)
        model.eval()
        return model


if __name__ == '__main__':
    yolov3 = Yolov3('./config/yolov3.cfg', './weights/yolov3.weights')
    img = Image.open('./data/samples/dog.jpg')
    # img = functional.pil_to_tensor(img) / 255.0
    output = yolov3.default_predictor(img)
    img = yolov3.visual_instance_predictions(img, output, mode='pil')
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
    # a = RetinaNet()
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
