from __future__ import absolute_import

from train_yolo_attack2 import *

import os
import setproctitle

setproctitle.setproctitle("train_adversarial_patch_coronaPolvo")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        # sys.path.append('/home/corona/attack/Fooling-Object-Detection-Network')
    trainer = PatchTrainer()
    load_from_file='/home/disk2/ray/workspace/Fly_Pluche/random_patch.png'
    is_random=False
    trainer.train(load_from_file,is_random)
