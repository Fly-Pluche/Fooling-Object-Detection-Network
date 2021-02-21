import torch
from detectron2.engine.defaults import DefaultPredictor


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super(Predictor, self).__init__(cfg)

    def __call__(self, image):
        """
        image: torch tensor [3,416,416]
        """
        height = image.shape[1]
        width = image.shape[2]
        image = image * 255
        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions
