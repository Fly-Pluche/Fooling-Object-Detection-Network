import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from patch_config import patch_configs
from patch import PatchTransformer
from patch import PatchApplier
from torch.utils.data import DataLoader
from PIL import Image
from load_data import ListDataset
from torchvision.transforms import functional
from models import FasterRCNN

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)


class PatchVisual(nn.Module):
    def __init__(self):
        super(PatchVisual, self).__init__()
        self.config = patch_configs['base']()
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        datasets = ListDataset(self.config.txt_path)
        data_loader = DataLoader(
            datasets,
            batch_size=1,
            num_workers=8,
            shuffle=True,
        )
        self.data = iter(data_loader)

    def forward(self, patch_path, data):
        """
        visual adv patch on the datasets
        return: one image which is added patch
        """
        # load path image and turn it to torch's format
        adv_patch = Image.open(patch_path)
        adv_patch = functional.pil_to_tensor(adv_patch).cuda()
        adv_patch = adv_patch / 255.
        # load a.json image information
        image, boxes, label = data
        image = image.cuda()
        boxes = boxes.cuda()
        label = label.cuda()
        # produce adv patch mask and apply it on the image
        adv_patch_t = self.patch_transformer(adv_patch, boxes, label)
        image = self.patch_applier(image, adv_patch_t)
        image = image[0].detach().cpu()
        image = np.asarray(functional.to_pil_image(image))
        return image


if __name__ == '__main__':
    visual = PatchVisual()
    data = next(visual.data)
    data = next(visual.data)
    data = next(visual.data)
    data = next(visual.data)
    image = np.asarray(functional.to_pil_image(data[0][0].detach().cpu()))
    image_attack = visual('/home/corona/attack/Fooling-Object-Detection-Network/patches/32.jpg', data)
    image_gray_patch = visual('./patches/gray.jpg', data)
    model = FasterRCNN()
    output1 = model.default_predictor_(image)
    output2 = model.default_predictor_(image_attack)
    output3 = model.default_predictor_(image_gray_patch)
    visual_image1 = model.visual_instance_predictions(image, output1)
    visual_image2 = model.visual_instance_predictions(image_attack, output2)
    visual_image3 = model.visual_instance_predictions(image_gray_patch, output3)

    plt.subplot(2, 2, 1)
    plt.title('original image')
    plt.imshow(visual_image1[:, :, ::-1])

    plt.subplot(2, 2, 2)
    plt.title('adv image')
    plt.imshow(visual_image2[:, :, ::-1])

    plt.subplot(2, 2, 3)
    plt.title('gray patch image')
    plt.imshow(visual_image3[:, :, ::-1])
    plt.show()
    # img = torch.full((3, 300, 300), 0.4)
    # img = np.asarray(transforms.ToPILImage()(img)) * 255
    # cv2.imwrite('./patches/gray.jpg', img)
