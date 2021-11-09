"""
对比使用频率loss后，保存为JPG图片对于攻击效果的影响
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from evaluator import *
from utils.utils import generate_patch

PATCH_PATH = [
    # 普通的没有使用频率loss
    '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211014-214115_base_YOLO_with_coco_datasets2/91.7_asr.png',
    '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211014-214115_base_YOLO_with_coco_datasets2/91.7_asr.jpg',
    '/home/disk2/ray/workspace/Fly_Pluche/attack/Experiment_logs/TV_LOSS/608/20211005-104842_base_YOLO_with_coco_datasets2/90.8_asr.png',
    '/home/disk2/ray/workspace/Fly_Pluche/attack/Experiment_logs/TV_LOSS/608/20211005-104842_base_YOLO_with_coco_datasets2/90.8_asr.jpg',
]


def run():
    config = patch_configs['base']()  # load base config
    model = Yolov3(config.model_path, config.model_image_size, config.classes_path)
    model.set_image_size(config.img_size[0])
    test_data = DataLoader(
        ListDataset(config.coco_val_txt),
        num_workers=16,
        batch_size=config.batch_size,
        drop_last=True
    )
    asr_calculate = ObjectVanishingASR(config.img_size, use_deformation=False)
    asr_calculate.register_dataset(model, test_data)
    patch_evaluator = PatchEvaluator(model, test_data, use_deformation=False).cuda()
    with open('./outputs.txt', 'w') as f:
        pass
    for patch_path in PATCH_PATH:
        adv_path_cpu = generate_patch(config, load_from_file=patch_path)
        # eval patch
        with torch.no_grad():
            # calculate ap50
            ap = patch_evaluator(adv_path_cpu, 0.5)
            # calculate asr50
            asr_calculate.inference_on_dataset(adv_path_cpu, clean=False)
            ASR, ASRs, ASRm, ASRl = asr_calculate.calculate_asr()
        with open('./outputs.txt', 'a') as f:
            f.write('AP,ASR,ASRs,ASRm,ASRl\n')
            f.write(f'{ap},{ASR},{ASRs},{ASRm},{ASRl}\n')
        print('ap:', ap)
        print('ASR:', ASR)
        print('ASRs:', ASRs)
        print('ASRm', ASRm)
        print('ASRl', ASRl)


if __name__ == '__main__':
    run()

    # import cv2
    # img = cv2.imread(
    #     '/home/disk2/ray/workspace/Fly_Pluche/attack/Experiment_logs/TV_LOSS/608/20211005-104842_base_YOLO_with_coco_datasets2/90.8_asr.png')
    # cv2.imwrite(
    #     '/home/disk2/ray/workspace/Fly_Pluche/attack/Experiment_logs/TV_LOSS/608/20211005-104842_base_YOLO_with_coco_datasets2/90.8_asr.jpg',
    #     img)
