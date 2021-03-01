from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files
    """

    def __init__(self):
        """
        set the defaults
        """
        self.img_dir = "/home/corona/datasets/flir_yolo/train/images"
        self.lab_dir = "/home/corona/datasets/flir_yolo/train/labels"
        self.config_file = "/home/corona/attack/PaperCode/configs/yolo_person.cfg"
        self.weight_file = "/home/corona/attack/PyTorch-YOLOv3/good_weights/yolov3_ckpt_98.pth"
        self.txt_path = '/home/corona/datasets/flir_yolo/train/train.txt'
        self.save_adv_patch_path = '/home/corona/attack/PaperCode2/patches'
        self.patch_size = 300

        self.start_learning_rate = 0.0025

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)

        self.batch_size = 1

        self.loss_target = lambda obj, cls: obj * cls
        # width height
        self.img_size = (416, 416)
        # the number of gauss function
        self.gauss_num = 20


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
