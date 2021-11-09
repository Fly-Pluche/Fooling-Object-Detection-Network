from patch_config import *
import torch
from torch.utils.data import DataLoader
from models import *
from load_data import ListDataset
from torchvision import transforms
from evaluator import PatchEvaluator
from patch import PatchTransformerPro, PatchApplierPro, PatchApplier, PatchTransformer
from tools import save_predict_image_torch
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def generate_patch(config, load_from_file=None, is_random=False, is_cmyk=0):
    # load a image from local patch
    if load_from_file is not None:
        patch = Image.open(load_from_file)
        patch = patch.resize((config.patch_size, config.patch_size))
        patch = transforms.PILToTensor()(patch) / 255.
        return patch
    if is_random:
        if is_cmyk:
            return torch.rand((4, config.patch_size, config.patch_size))
        else:
            return torch.rand((3, config.patch_size, config.patch_size))
    if is_cmyk:
        return torch.full((4, config.patch_size, config.patch_size), 0.5)
    else:
        return torch.full((3, config.patch_size, config.patch_size), 0.5)


def save_images_in_visual(patch_path, save_path):
    """
    use for result of visual
    :param patch: the attack patch path
    :param save_path: the visual of result save path
    :return: NULL
    """
    config = patch_configs['base']()
    model = Yolov3(config.model_path, config.model_image_size, config.classes_path)
    model.set_image_size(config.img_size[0])
    test_data = DataLoader(
        ListDataset(config.coco_val_txt, number=2000),
        num_workers=16,
        batch_size=config.batch_size
    )
    # './logs/20211001-153330_base_YOLO_with_coco_datasets2/86.6_asr.png'
    patch = generate_patch(config,
                           load_from_file=patch_path,
                           is_cmyk=False)
    with torch.no_grad():
        for id, item in enumerate(tqdm(test_data)):
            image_batch, people_boxes, labels_batch = item
            people_boxes = people_boxes.cuda()
            labels_batch = labels_batch.cuda()
            image_batch = image_batch.cuda()
            adv_patch = patch.cuda()
            patch_transformer = PatchTransformer().cuda()
            patch_applier = PatchApplier().cuda()
            adv_batch_t = patch_transformer(adv_patch, people_boxes, labels_batch)
            p_img_batch = patch_applier(image_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch, (config.img_size[0], config.img_size[1]),
                                        mode='bilinear')
            images = torch.unbind(p_img_batch, dim=0)
            for idx, image in enumerate(images):
                path = os.path.join(save_path, f"{id}_{idx}.png")
                save_predict_image_torch(model, image, path)
            # print("运行结束")


if __name__ == '__main__':
    save_images_in_visual("./logs/20211001-153330_base_YOLO_with_coco_datasets2/86.6_asr.png",
                          "./save_visual_images/")
