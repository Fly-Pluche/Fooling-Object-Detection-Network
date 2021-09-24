from __future__ import absolute_import

import time
import logging
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from asr import ObjectVanishingASR
from evaluator import *
from evaluator import PatchEvaluator
from load_data import ListDatasetAnn
from models import *
import matplotlib.pyplot as plt
from patch import *
from patch_config import *
from utils.transforms import CMYK2RGB
from utils.utils import *
import warnings

warnings.filterwarnings('ignore')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)
config = patch_configs['base']()
test_data = DataLoader(
    ListDataset(config.coco_val_txt, number=2000),
    num_workers=16,
    batch_size=config.batch_size
)

model_ = Yolov3(config.model_path, config.model_image_size, config.classes_path)
model_.set_image_size(config.img_size[0])
patch_evaluator = PatchEvaluator(model_, test_data, use_deformation=False).cuda()
patch = Image.open('./logs/20210915-174930_base_YOLO_with_coco_datasets2/84.1_asr.png')
patch = patch.resize((config.patch_size, config.patch_size))
adv_patch_cpu = transforms.PILToTensor()(patch) / 255.
patch_evaluator.save_visual_images(adv_patch_cpu.clone(), './images/yolo_output', 1)
