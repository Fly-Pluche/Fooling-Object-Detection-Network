from useful_imports import *

config = patch_configs['base']()
# model = FasterRCNN()
model = Yolov3(config.model_path, config.model_image_size, config.classes_path)
model.set_image_size(config.img_size[0])

test_data = DataLoader(
    ListDataset(config.coco_val_txt),
    num_workers=16,
    batch_size=8,
    drop_last=True
)
asr_calculate = ObjectVanishingASR(config.img_size, use_deformation=False)
asr_calculate.register_dataset(model, test_data)

path = '/home/disk2/ray/workspace/Fly_Pluche/attack/logs/20211203-170438_base_八角 mask_FFT_隐式训练-18step-lr=0.005 ycbcr[finial-fix2]/91.3_asr.pt'
adv_path = torch.load(path)
asr_calculate.inference_on_dataset(adv_path, clean=False)
ASR, ASRs, ASRm, ASRl = asr_calculate.calculate_asr()
print(ASR, ASRl, ASRm, ASRs)
