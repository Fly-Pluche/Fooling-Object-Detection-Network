from torch import optim
from models import *
import configparser

class BaseConfig(object):
    """
    Default parameters for all config files
ROOT_PATH = /home/mist/deepfooling
IMG_SIZE = 1000
IMG_SIZE_BIG = 1500
LOG_PATH = /home/mist/logs
SAVE_ADV_PATCH_PATH = /home/mist/deepfooling/train.txt
DEEPFOOLING_TXT = /home/mist/deepfooling
PATH_SIZE = 500
    """

    def __init__(self):
        """
        set the defaults
        """
        config = configparser.ConfigParser()
        config.read('./config/mist.cfg')
        self.config_file = "/home/corona/attack/PaperCode/configs/yolo_person.cfg"
        self.weight_file = "/home/corona/attack/PyTorch-YOLOv3/good_weights/yolov3_ckpt_98.pth"
        self.txt_path = '/home/corona/datasets/WiderPerson/train/train2.txt'
        self.save_adv_patch_path = config['default']['SAVE_ADV_PATCH_PATH']
        self.deepfooling_txt = config['default']['DEEPFOOLING_TXT']
        self.patch_size = int(config['default']['PATH_SIZE'])
        self.root_path = config['default']['ROOT_PATH']
        self.start_learning_rate = 0.007
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.batch_size = int(config['default']['BATCH_SIZE'])
        # width height
        self.img_size = (int(config['default']['IMG_SIZE']), int(config['default']['IMG_SIZE']))
        self.img_size_big = (int(config['default']['IMG_SIZE_BIG']), int(config['default']['IMG_SIZE_BIG']))
        # anchor base
        self.anchor_base = True

        # the number of gauss function
        self.gauss_num = 20
        self.max_lab = 3
        self.log_path = config['default']['LOG_PATH']


patch_configs = {
    "base": BaseConfig,
}
