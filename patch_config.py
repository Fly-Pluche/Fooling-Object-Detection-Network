from torch import optim
from models import *


class BaseConfig(object):
    """
    Default parameters for all config files
    """

    def __init__(self):
        """
        set the defaults
        """

        self.config_file = "/home/corona/attack/PaperCode/configs/yolo_person.cfg"
        self.weight_file = "/home/corona/attack/PyTorch-YOLOv3/good_weights/yolov3_ckpt_98.pth"
        self.txt_path = '/home/corona/datasets/WiderPerson/train/train2.txt'
        self.save_adv_patch_path = '/home/corona/attack/Fooling-Object-Detection-Network/patches'
        self.deepfooling_txt = '/home/corona/deepfooling/train.txt'
        self.patch_size = 500
        self.root_path = '/home/corona/deepfooling'
        self.start_learning_rate = 0.007
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.batch_size = 4
        # width height
        self.img_size = (1000, 1000)
        self.img_size_big = self.img_size
        # anchor base
        self.anchor_base = True

        # the number of gauss function
        self.gauss_num = 20
        self.max_lab = 3
        self.log_path = '/home/corona/attack/Fooling-Object-Detection-Network/logs'


patch_configs = {
    "base": BaseConfig,
}
