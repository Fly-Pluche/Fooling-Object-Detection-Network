from __future__ import absolute_import
import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import _create_text_labels
from torchvision import transforms


class PatchVisualizer(Visualizer):
    def __init__(self, img_tensor, model, threshold=0.5):
        img_rgb = self.tensor2rgb(img_tensor)
        super(PatchVisualizer, self).__init__(img_rgb[:, :, ::-1], model.cfg.DATASETS.TRAIN[0])
        self.threshold = threshold

    def tensor2rgb(self, img):
        """
        turn a tensor float image to a uint8 rgb image
        Args:
            img: a torch tensor [3,w,h] float type. The image pixel values are between 0 and 1.
        return:
            a rgb numpy array
        """
        img = transforms.ToPILImage()(img)
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
