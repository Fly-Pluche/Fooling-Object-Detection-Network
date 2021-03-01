from __future__ import absolute_import
import os
import cv2
import time
import torch
import numpy as np
from patch_config import *
from models import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from load_data import ListDataset
from torchvision import transforms
from patch import PatchTransformer, PatchApplier, PatchGauss
from evaluator import MaxProbExtractor, TotalVariation, PatchEvaluator
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)


class PatchTrainer(object):
    def __init__(self):
        super(PatchTrainer, self).__init__()
        self.config = patch_configs['base']()  # load base config
        self.model_ = RetinaNet()
        self.writer = self.init_tensorboard(name='base')
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()
        self.max_prob_extractor = MaxProbExtractor().cuda()
        self.total_variation = TotalVariation().cuda()
        self.patch_gauss = PatchGauss().cuda()

    def init_tensorboard(self, name=None):
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'/home/corona/attack/Fooling-Object-Detection-Network/logs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        optimizer a adversarial patch
        """
        # load train datasets
        datasets = ListDataset(self.config.txt_path)
        train_data = DataLoader(
            datasets,
            batch_size=self.config.batch_size,
            num_workers=8,
            shuffle=True,
        )

        epoch_length = len(train_data)
        # generate a gray patch
        adv_gauss_cpu = self.generate_gauss()
        adv_gauss_cpu.requires_grad_(True)

        # use adam to update adv_patch
        optimizer = torch.optim.Adam([adv_gauss_cpu], lr=self.config.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)  # used to update learning rate

        # create a evaluator for adv patch
        attack_evaluator = PatchEvaluator(self.model_, train_data)

        for epoch in range(1000):
            ep_det_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            i_batch = 0
            for image_batch, boxes_batch, labels_batch in tqdm(train_data):
                i_batch += 1
                image_batch = image_batch.cuda()
                labels_batch = labels_batch.cuda()
                boxes_batch = boxes_batch.cuda()
                adv_gauss = adv_gauss_cpu.cuda()
                adv_patch = self.patch_gauss(adv_gauss)
                ap50 = attack_evaluator(adv_patch)
                print('===='*100)
                print(ap50)
                print('===='*100)
                adv_batch_t = self.patch_transformer(adv_patch, boxes_batch, labels_batch)
                p_img_batch = self.patch_applier(image_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.config.img_size[1], self.config.img_size[0]))

                # img = p_img_batch[1, :, :, ]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img = np.asarray(img)
                # plt.imshow(img)
                # plt.show()
                # exit()

                max_prob = self.max_prob_extractor(self.model_, p_img_batch)


                tv = self.total_variation(adv_patch)

                tv_loss = tv
                det_loss = torch.sum(max_prob ** 2)

                loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                ep_det_loss += det_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_gauss_cpu.data.clamp_(0, 1)  # keep

                if det_loss.detach().cpu().numpy() < 0.8:
                    img = np.asarray(transforms.ToPILImage()(adv_patch.detach().cpu()))
                    name = './patches/' + str(epoch) + '_' + str(i_batch) + '_' + str(int(det_loss * 10)) + '.jpg'
                    cv2.imwrite(name, img)

                if i_batch % 5 == 0:
                    iteration = epoch_length * epoch + i_batch
                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('misc/epoch', epoch, iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                    self.writer.add_image('patch', adv_patch.cpu(), iteration)

                if i_batch + 1 >= len(train_data):
                    print('\n')
                else:
                    # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    del adv_batch_t, max_prob, det_loss, p_img_batch, tv_loss, loss
                    torch.cuda.empty_cache()

            # save adversarial patch
            # with torch.no_grad():
            #     adv_patch_save = self.patch_gauss(adv_gauss_cpu.cuda())
            #     adv_patch_save = transforms.ToPILImage()(adv_patch_save.detach().cpu())
            #     adv_patch_save = np.asarray(np.uint8(adv_patch_save))
            #     cv2.imwrite(os.path.join(self.config.save_adv_patch_path, str(epoch) + '.jpg'), adv_patch_save)
            # evaluate the model after attacking

            ep_det_loss = ep_det_loss / len(train_data)
            ep_tv_loss = ep_tv_loss / len(train_data)
            ep_loss = ep_loss / len(train_data)

            scheduler.step(ep_loss)

            # adv_gauss = adv_gauss_cpu.cuda()
            # adv_patch = self.patch_gauss(adv_gauss)
            # result = attack_evaluator(adv_patch)
            # print(result)

            if True:
                self.writer.add_scalar('ep_det_loss', ep_det_loss, epoch)
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('   TV LOSS: ', ep_tv_loss)
            # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            del adv_batch_t, max_prob, det_loss, p_img_batch, tv_loss, loss
            torch.cuda.empty_cache()

    def generate_patch(self):
        return torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)

    def generate_gauss(self):
        return torch.rand(2, self.config.gauss_num)
