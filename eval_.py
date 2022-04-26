from useful_imports import *

config = patch_configs['base']()
# model = RetinaNet()
model = Yolov3(config.model_path, config.model_image_size, config.classes_path)
model.set_image_size(config.img_size[0])

test_data = DataLoader(
    ListDataset(config.coco_train_txt, enhancd=False),
    num_workers=16,
    batch_size=8,
    drop_last=True,
    shuffle=False
)

asr_calculate = ObjectVanishingASR(config.img_size, use_deformation=False)
asr_calculate.register_dataset(model, test_data)
#
# path = '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211203-170438_base_八角 mask_FFT_隐式训练-18step-lr=0.005 ycbcr[finial-fix2]/91.3_asr.pt'
# path = '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211204-081549_base_八角 mask_FFT_隐式训练-18step-lr=0.005 ycbcr[finial-fix2-keep]/86.8_asr.pt'
# path = '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211205-111134_base_八角 mask_FFT_隐式训练-18step-lr=0.005 rgb[finial-pre-7]/85.3_asr.pt'
# path = '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211129-222521_base_八角 origin/87.9_asr.pt'
# adv_path = generate_patch(config, is_random=True).cuda()
# adv_path = torch.load(path)
# functional.to_pil_image(adv_path).save('./final_images/final_ycbcr.jpg')
#
# functional.to_pil_image(adv_path).save('./final_images/final_ycbcr.png')

# asr_calculate.inference_on_dataset(adv_path, clean=False)
# ASR, ASRs, ASRm, ASRl = asr_calculate.calculate_asr()
# print(ASR, ASRl, ASRm, ASRs)
paths = [
    '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211203-170438_base_八角 mask_FFT_隐式训练-18step-lr=0.005 ycbcr[finial-fix2]/91.3_asr.pt',
    '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211204-081549_base_八角 mask_FFT_隐式训练-18step-lr=0.005 ycbcr[finial-fix2-keep]/86.8_asr.pt',
    '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211205-111134_base_八角 mask_FFT_隐式训练-18step-lr=0.005 rgb[finial-pre-7]/85.3_asr.pt',
    '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211129-222521_base_八角 origin/87.9_asr.pt'
]
floders = ['final-ycbcr', 'ycbcr-only', 'rgb_pre', 'origin']
patch_evaluator = PatchEvaluator(model, test_data, use_deformation=False).cuda()
adv_path = generate_patch(config, is_random=True).cuda()
patch_evaluator.save_all_test(adv_path, './visual_images2/gray', clean=False)
patch_evaluator.save_all_test(adv_path, './visual_images2/clean', clean=True)
for path, floder in zip(paths, floders):
    adv_path = torch.load(path)
    patch_evaluator.save_all_test(adv_path, f'./visual_images2/{floder}')
    del adv_path
#
# patch_evaluator.save_all_test(adv_path, './visual_images/ycbcr-only')
