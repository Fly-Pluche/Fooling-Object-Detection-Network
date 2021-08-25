from __future__ import absolute_import

from train_patch import *
from models import *
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
trainer = PatchTrainer()
trainer.train()