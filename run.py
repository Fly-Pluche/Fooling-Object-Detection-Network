from __future__ import absolute_import

from train_patch_Pro import *
from models import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append('/home/corona/attack/Fooling-Object-Detection-Network')

    trainer = PatchTrainer()
    trainer.train()
