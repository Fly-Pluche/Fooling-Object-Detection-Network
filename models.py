from __future__ import absolute_import
import detectron2
import cv2
import os
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from torch import nn
from utils.img_tools import ImageTools
from predict import Predictor
from utils.visualizer import Visualizer_

class BaseModel(nn.Module):
    def __init__(self, model):
        super(BaseModel, self).__init__()
        self.output = None
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        self.predictor = Predictor(self.cfg)
        self.default_predictor = DefaultPredictor(self.cfg)
        self.model = self.predictor.model

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
        return self.default_predictor(img)

    def visual_instance_predictions(self, img, output,mode='tensor'):
        """
        draw instance boxes on the image
        """
        v = Visualizer_(img, self,mode=mode)
        out = v.draw_instance_predictions(output["instances"].to('cpu'))
        return out.get_image()[:, :, ::-1]

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


if __name__ == '__main__':
    import time

    t = ImageTools()
    faster = FasterRCNN()
    im = cv2.imread('/home/corona/datasets/flir_yolo/train/images/06478.jpeg')
    time1 = time.time()
    outputs = faster(im)["instances"]
    print(time.time() - time1)
    classes = outputs.pred_classes
    scores = outputs.scores
    print(classes)
    print(scores[classes == 0])
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(faster.cfg.DATASETS.TRAIN[0]), scale=1)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # out = out.get_image()[:, :, ::-1]
    # t.cv2_imshow(out, transform=False)
