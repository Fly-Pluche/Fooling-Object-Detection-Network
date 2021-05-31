from __future__ import absolute_import
import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import _create_text_labels
from torchvision.transforms import functional
from detectron2.data import MetadataCatalog


class Visualizer_(Visualizer):
    def __init__(self, img, model, threshold=0.5, mode='tensor'):
        """
        Args:
            img: one image (support tensor and rgb np array)
            model: detectron2 model
            threshold: boxes which confidence score low than threshold will not be drawn
        """
        if mode == 'tensor':
            img_rgb = self.tensor2rgb(img)
        else:
            img_rgb = img
        super(Visualizer_, self).__init__(img_rgb, MetadataCatalog.get(model.cfg.DATASETS.TRAIN[0]))
        self.threshold = threshold

    def tensor2rgb(self, img):
        """
        turn a tensor float image to a uint8 rgb image
        Args:
            img: a torch tensor [3,w,h] float type. The image pixel values are between 0 and 1.
        return:
            a rgb numpy array
        """
        img = functional.to_pil_image(img)
        img = np.asarray(img)
        return img

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.

        corona: rewrite this function in detectron2. This function will only draw boxes on the image
        """

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        boxes = boxes[scores >= self.threshold]
        classes = classes[scores >= self.threshold]
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        alpha = 0.5

        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            alpha=alpha,
        )

        return self.output
