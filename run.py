from __future__ import absolute_import

from train_patch import *

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append('/home/corona/attack/Fooling-Object-Detection-Network')
    trainer = PatchTrainer()
    trainer.train()
