from __future__ import absolute_import

from train_yolo_attack import *

import os
import setproctitle

setproctitle.setproctitle("train_adversarial_patch_coronaPolvo")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        # sys.path.append('/home/corona/attack/Fooling-Object-Detection-Network')
    trainer = PatchTrainer()
    trainer.train()
