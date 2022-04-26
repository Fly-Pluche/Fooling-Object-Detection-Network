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
        config.read('./config/local2.cfg')
        self.config_file = "/home/corona/attack/PaperCode/configs/yolo_person.cfg"
        self.weight_file = "/home/corona/attack/PyTorch-YOLOv3/good_weights/yolov3_ckpt_98.pth"
        self.txt_path = '/home/corona/datasets/WiderPerson/train/train2.txt'
        self.save_adv_patch_path = config['DEFAULT']['SAVE_ADV_PATCH_PATH']
        self.step_size = float(config['DEFAULT']['STEP_SIZE'])
        self.gamma = float(config['DEFAULT']['GAMMA'])
        self.deepfooling_txt = config['DEFAULT']['DEEPFOOLING_TXT']
        self.patch_size = int(config['DEFAULT']['PATH_SIZE'])
        self.root_path = config['DEFAULT']['ROOT_PATH']
        # self.start_learning_rate = 0.004
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.batch_size = int(config['DEFAULT']['BATCH_SIZE'])
        # width height
        self.img_size = (int(config['DEFAULT']['IMG_SIZE']), int(config['DEFAULT']['IMG_SIZE']))
        self.img_size_big = self.img_size
        # the number of gauss function
        self.gauss_num = 20
        self.max_lab = 3
        self.log_path = config['DEFAULT']['LOG_PATH']
        self.is_cmyk = int(config['DEFAULT']['IS_CMYK'])
        self.optim = config['DEFAULT']['OPTIM']
        self.fft_size = float(config['DEFAULT']['FFT_SIZE'])
        self.patch_scale = float(config['DEFAULT']['PATCH_SCALE'])
        if self.optim == 'adam':
            self.start_learning_rate = float(config['DEFAULT']['START_LEARNING_RATE'])
        else:
            self.start_learning_rate = 32 / 255.
        self.model_path = config['DEFAULT']['MODEL_PATH']
        self.model_image_size = int(config['DEFAULT']['MODEL_IMAGE_SIZE'])
        self.classes_path = config['DEFAULT']['CLASSES_PATH']
        self.detail_info = config['DEFAULT']['INFO']
        self.coco_train_txt = config['DEFAULT']['COCO_TRAIN']
        self.coco_val_txt = config['DEFAULT']['COCO_VAL']


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'


class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"


class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls


class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj
}
