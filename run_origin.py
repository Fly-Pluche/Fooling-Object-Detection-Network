from __future__ import absolute_import

from train_yolo_attack_origin import *

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
    load_mask_from_file = '/home/disk2/ray/workspace/Fly_Pluche/random_mask.png'
    load_patch_from_file = '/home/disk2/ray/workspace/Fly_Pluche/random_patch.png'
    # load_from_file='/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211104-163254_base_八角 mask_FFT 有frequency loss/85.6_asr.png'

    is_random = False
    trainer.train(load_patch_from_file, load_mask_from_file, is_random)
