from __future__ import absolute_import

import time

import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from evaluator import TotalVariation, UnionDetectorBCE, ConfMaxExtractor, MaxExtractor
from evaluator import PatchEvaluator
from load_data import ListDatasetAnn
from models import *
import matplotlib.pyplot as plt
from patch import PatchTransformerPro, PatchApplierPro
from patch_config import *
from utils.transforms import CMYK2RGB
from utils.utils import pytorch_imshow
import warnings

warnings.filterwarnings('ignore')

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)


class PatchTrainer(object):
    def __init__(self):
        super(PatchTrainer, self).__init__()
        self.config = patch_configs['base']()  # load base config
        # self.model_ = Yolov3(img_size=self.config.img_size[0])
        # self.name = 'Yolov3'
        self.model_ = FasterRCNN()
        self.name = 'FasterRCNN'
        self.log_path = self.config.log_path
        self.writer = self.init_tensorboard(name='base')
        os.mkdir(os.path.join(self.log_path, 'visual_image'))
        self.image_save_path = os.path.join(self.log_path, 'visual_image')
        self.patch_transformer = PatchTransformerPro().cuda()
        self.patch_applier = PatchApplierPro().cuda()
        self.max_extractor = MaxExtractor().cuda()
        self.total_variation = TotalVariation().cuda()
        self.union_detector = UnionDetectorBCE().cuda()
        self.patch_evaluator = None
        self.entropy = nn.BCELoss()
        self.is_cmyk = self.config.is_cmyk

    def init_tensorboard(self, name=None):
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.log_path = f'{self.config.log_path}/{time_str}_{name}_{self.name}'
            return SummaryWriter(f'{self.config.log_path}/{time_str}_{name}_{self.name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        optimizer a adversarial patch
        """
        # load train datasets
        datasets = ListDatasetAnn(self.config.deepfooling_txt, range_=[80, 780])
        train_data = DataLoader(
            datasets,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=True,
        )
        test_data = DataLoader(
            ListDatasetAnn(self.config.deepfooling_txt, range_=[0, 80]),
            num_workers=4,
            batch_size=self.config.batch_size
        )
        self.patch_evaluator = PatchEvaluator(self.model_, test_data)

        epoch_length = len(train_data)

        # generate a rgb patch
        # adv_patch_cpu = self.generate_patch(
        # load_from_file='logs/20210810-113457_base_RetinaNet_without_iou_loss/71.3587253925269.jpg')
        adv_patch_cpu = self.generate_patch(is_random=True, is_cmyk=self.is_cmyk)
        adv_patch_cpu.requires_grad_(True)
        optimizer = torch.optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)  # used to update learning rate

        # Parameters to initialize some dynamically updated weights
        N = 4
        w_iou = 1
        w_conf_union = 1
        w_tv = 1
        w_union_attack = 1
        last_iou_loss = -1
        last_conf_union_loss = -1
        last_tv_loss = -1
        last_union_attack_loss = -1

        # start training 
        min_ap = 1
        for epoch in range(10000):
            ep_iou_all = 0
            ep_conf_union_loss = 0
            ep_union_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            i_batch = 0
            for image_batch, clothes_boxes_batch, people_boxes_batch, labels_batch, landmarks_batch, segmentations_batch in tqdm(
                    train_data):
                i_batch += 1

                image_batch = image_batch.cuda()
                labels_batch = labels_batch.cuda()
                people_boxes_batch = people_boxes_batch.cuda()
                adv_patch = adv_patch_cpu.cuda()

                if self.is_cmyk:
                    adv_patch = CMYK2RGB(adv_patch)

                # Attach the attack image to the clothing
                adv_batch_t, adv_batch_mask_t = self.patch_transformer(adv_patch,
                                                                       clothes_boxes_batch,
                                                                       segmentations_batch,
                                                                       landmarks_batch,
                                                                       image_batch)
                p_img_batch = self.patch_applier(image_batch, adv_batch_t, adv_batch_mask_t)
                p_img_batch = F.interpolate(p_img_batch, (self.config.img_size[1], self.config.img_size[0]))

                # calculate each part of the loss
                conf_loss_union_image, iou_loss = self.max_extractor(self.model_, p_img_batch, people_boxes_batch)
                tv_loss = self.total_variation(adv_patch)
                predicted_id, attack_id = self.union_detector(self.model_, image_batch, p_img_batch, people_boxes_batch)
                union_attack_loss = self.entropy(predicted_id, attack_id)

                print()
                print('iou_loss: ', iou_loss)
                print('conf_loss_union_image', conf_loss_union_image)
                print("union_attack_loss", union_attack_loss)

                # Dynamic update weight parameter
                with torch.no_grad():
                    if last_iou_loss != -1:
                        r_conf_single = iou_loss / last_iou_loss
                        r_conf_union = conf_loss_union_image / last_conf_union_loss
                        r_tv = tv_loss / last_tv_loss
                        r_union = union_attack_loss / last_union_attack_loss
                        rs = torch.stack([r_conf_single, r_conf_union, r_tv, r_union])
                        rs = torch.exp(rs)
                        sum_rs = torch.sum(rs)
                        w_iou = (N * rs[0]) / sum_rs
                        w_conf_union = N * rs[1] / sum_rs
                        w_tv = N * rs[2] / sum_rs
                        w_union_attack = N * rs[3] / sum_rs

                loss = w_iou * iou_loss + w_conf_union * conf_loss_union_image + w_tv * tv_loss + w_union_attack * union_attack_loss
                # loss = iou_loss
                # loss = conf_loss_union_image
                # loss = conf_loss_union_image
                # loss = union_attack_loss

                # update last loss
                last_iou_loss = iou_loss.detach()
                last_conf_union_loss = conf_loss_union_image.detach()
                last_tv_loss = tv_loss.detach()
                last_union_attack_loss = union_attack_loss.detach()

                # evaluate
                ep_iou_all += iou_loss.detach().cpu().numpy()
                ep_conf_union_loss += conf_loss_union_image.detach().cpu().numpy()
                ep_union_loss += union_attack_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                adv_patch_cpu.grad = torch.sign(adv_patch_cpu.grad)
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                if i_batch % 23 == 0:
                    iteration = epoch_length * epoch + i_batch
                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/iou_loss', iou_loss.detach().cpu().numpy(),
                                           iteration)
                    self.writer.add_scalar('loss/conf_loss_union_image', conf_loss_union_image.detach().cpu().numpy(),
                                           iteration)
                    self.writer.add_scalar('loss/union_loss', union_attack_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_image('patch', adv_patch.cpu(), iteration)
                    # self.writer.add_image('patch', adv_patch.detach().cpu(), iteration)

                del adv_batch_t, p_img_batch, adv_batch_mask_t, conf_loss_union_image, iou_loss, tv_loss,
                torch.cuda.empty_cache()

            # visual attack effection
            if epoch % 10 == 0:
                self.patch_evaluator.save_visual_images(adv_patch_cpu.detach().clone(), self.image_save_path, epoch)

            # eval patch
            with torch.no_grad():
                if self.is_cmyk:
                    adv_cmyk = adv_patch_cpu.clone()
                    adv = adv_cmyk.cuda()
                    adv = CMYK2RGB(adv)
                else:
                    adv = adv_patch_cpu.clone()

                # calculate ap50
                ap = self.patch_evaluator(adv, 0.5)
                self.writer.add_scalar('ap', ap, epoch)

                # save better patch image
                if float(ap) < min_ap:
                    name = os.path.join(self.log_path, str(float(ap) * 100) + '.jpg')
                    torchvision.utils.save_image(adv, name)
                    min_ap = float(ap)
                    if self.is_cmyk:
                        name2 = os.path.join(self.log_path, str(float(ap) * 100) + '.TIF')
                        adv_cmyk = functional.to_pil_image(adv_cmyk)
                        adv_cmyk.save(name2)

            ep_iou_all = ep_iou_all / len(train_data)
            ep_conf_union_loss = ep_conf_union_loss / len(train_data)
            ep_union_loss = ep_union_loss / len(train_data)
            ep_tv_loss = ep_tv_loss / len(train_data)
            ep_loss = ep_loss / len(train_data)
            scheduler.step(ep_loss)

            if True:
                print('| EPOCH NR: ', epoch),
                print('| EPOCH LOSS: ', ep_loss)
                print("| AP: ", ap)
                self.writer.add_scalar('ep_iou_all', ep_iou_all, epoch)
                self.writer.add_scalar('ep_conf_union_loss', ep_conf_union_loss, epoch)
                self.writer.add_scalar('ep_union_loss', ep_union_loss, epoch)
                self.writer.add_scalar('ep_loss', ep_loss, epoch)
                self.writer.add_scalar('AP', ap, epoch)

            print('| ep_iou_all: ', ep_iou_all)
            print('| ep_tv_loss: ', ep_tv_loss)
            print('| ep_conf_union_loss:', ep_conf_union_loss)
            print('| ep_union_loss', ep_union_loss)
            # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            # del max_prob, det_loss, p_img_batch, tv_loss, loss, adv_batch_mask_t, iou_loss, union_iou_loss
            torch.cuda.empty_cache()

    def generate_patch(self, load_from_file=None, is_random=False, is_cmyk=0):
        # load a image from local patch
        if load_from_file is not None:
            patch = Image.open(load_from_file)
            patch = patch.resize((self.config.patch_size, self.config.patch_size))
            patch = transforms.PILToTensor()(patch) / 255.
            return patch
        if is_random:
            if is_cmyk:
                return torch.rand((4, self.config.patch_size, self.config.patch_size))
            else:
                return torch.rand((3, self.config.patch_size, self.config.patch_size))
        if is_cmyk:
            return torch.full((4, self.config.patch_size, self.config.patch_size), 0.5)
        else:
            return torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)


if __name__ == '__main__':
    a = PatchTrainer()
    img = a.generate_patch(is_random=False)
    img = functional.to_pil_image(img, mode='CMYK')
    img.save('./a.TIF')
