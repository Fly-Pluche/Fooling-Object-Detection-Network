from __future__ import absolute_import

import time

import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from evaluator import MaxExtractor, TotalVariation, UnionDetector
from evaluator import PatchEvaluator
from load_data import ListDatasetAnn
from models import *
from patch import PatchTransformerPro, PatchApplierPro, PatchTransformer
from patch_config import *

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)


class PatchTrainer(object):
    def __init__(self):
        super(PatchTrainer, self).__init__()
        self.config = patch_configs['base']()  # load base config
        self.model_ = FasterRCNN_R_101_FPN()
        self.name = 'Faster_RCNN_R101'
        self.config.anchor_base = True
        self.log_path = None
        self.writer = self.init_tensorboard(name='base')
        self.patch_transformer = PatchTransformerPro().cuda()
        self.patch_applier = PatchApplierPro().cuda()
        self.max_extractor = MaxExtractor().cuda()
        self.total_variation = TotalVariation().cuda()
        self.union_detector = UnionDetector().cuda()
        self.patch_evaluator = None
        self.entropy = nn.CrossEntropyLoss()

    def init_tensorboard(self, name=None):
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.log_path = f'/home/ma-user/work/logs/{time_str}_{name}_{self.name}'
            return SummaryWriter(f'/home/ma-user/work/logs/{time_str}_{name}_{self.name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        optimizer a adversarial patch
        """
        # load train datasets
        datasets = ListDatasetAnn(self.config.deepfooling_txt, range_=[160, 1760])
        train_data = DataLoader(
            datasets,
            batch_size=self.config.batch_size,
            num_workers=1,
            shuffle=True,
        )

        test_data = DataLoader(
            ListDatasetAnn(self.config.deepfooling_txt, range_=[0, 160]),
            num_workers=1,
            batch_size=self.config.batch_size
        )

        self.patch_evaluator = PatchEvaluator(self.model_, test_data)

        epoch_length = len(train_data)
        # generate a gray patch
        adv_patch_cpu = self.generate_patch(
            load_from_file='/home/corona/attack/Fooling-Object-Detection-Network/images/random_patch.jpg')
        adv_patch_cpu.requires_grad_(True)
        # use adam to update adv_patch
        optimizer = torch.optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)  # used to update learning rate
        min_ap = 1
        for epoch in range(10000):
            ep_det_loss = 0
            ep_tv_loss = 0
            ep_iou_loss = 0
            ep_union_iou_loss = 0
            ep_loss = 0
            i_batch = 0
            for image_batch, clothes_boxes_batch, people_boxes_batch, labels_batch, landmarks_batch, segmentations_batch in tqdm(
                    train_data):
                i_batch += 1
                image_batch = image_batch.cuda()
                labels_batch = labels_batch.cuda()
                people_boxes_batch = people_boxes_batch.cuda()
                adv_patch = adv_patch_cpu.cuda()
                adv_batch_t, adv_batch_mask_t = self.patch_transformer(adv_patch,
                                                                       clothes_boxes_batch,
                                                                       segmentations_batch,
                                                                       landmarks_batch,
                                                                       image_batch)
                p_img_batch = self.patch_applier(image_batch, adv_batch_t, adv_batch_mask_t)
                p_img_batch = F.interpolate(p_img_batch, (self.config.img_size[1], self.config.img_size[0]))
                # show apply affection
                # plt.imshow(np.array(functional.to_pil_image(p_img_batch[0])))
                # plt.show()
                # calculate union iou loss
                predicted_id, attack_id = self.union_detector(self.model_, image_batch, p_img_batch, people_boxes_batch)
                union_iou_loss = self.entropy(predicted_id, attack_id)
                max_prob, max_iou_t = self.max_extractor(self.model_, p_img_batch, people_boxes_batch)
                tv = self.total_variation(adv_patch)
                tv_loss = tv * 2
                # calculate the entropy
                #                 I_max_prob = torch.log2(1 / (1 - max_prob))
                print(max_prob)
                det_loss = torch.sum(max_prob)
                det_loss = det_loss
                # calculate iou loss
                iou_loss = torch.sum(max_iou_t)
                # calculate cross entropy of union detector
                print(det_loss)
                print(iou_loss)
                print(union_iou_loss)
                #                 loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) + iou_loss + union_iou_loss
                #                 loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) +
                #                 loss = det_loss
                loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) + iou_loss + union_iou_loss
                # evaluate
                ep_det_loss += det_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_iou_loss += iou_loss.detach().cpu().numpy()
                ep_union_iou_loss += union_iou_loss.detach().cpu().numpy()
                ep_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                if i_batch % 23 == 0:
                    iteration = epoch_length * epoch + i_batch
                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('union iou loss', union_iou_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('misc/epoch', epoch, iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                    self.writer.add_image('patch', adv_patch_cpu, iteration)

                if i_batch + 1 >= len(train_data):
                    print('\n')
                else:
                    # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    del adv_batch_t, max_prob, det_loss, p_img_batch, tv_loss, loss, adv_batch_mask_t, iou_loss, union_iou_loss
                    torch.cuda.empty_cache()

            # save adversarial patch
            # adv_patch_save = adv_patch_cpu.clone()
            # adv_patch_save = transforms.ToPILImage()(adv_patch_save.detach().cpu())
            # adv_patch_save = np.asarray(np.uint8(adv_patch_save))
            # cv2.imwrite(os.path.join(self.config.save_adv_patch_path, str(epoch) + '.jpg'), adv_patch_save)
            with torch.no_grad():
                adv = adv_patch_cpu.clone()
                ap = self.patch_evaluator(adv, 0.5)  # ap50
                self.writer.add_scalar('ap', ap, epoch)
                if float(ap) < min_ap:
                    name = os.path.join(self.log_path, str(float(ap) * 100) + '.jpg')
                    torchvision.utils.save_image(adv, name)
                    min_ap = float(ap)
            ep_det_loss = ep_det_loss / len(train_data)
            ep_tv_loss = ep_tv_loss / len(train_data)
            ep_iou_loss = ep_iou_loss / len(train_data)
            ep_loss = ep_loss / len(train_data)
            ep_union_iou_loss = ep_union_iou_loss / len(train_data)

            scheduler.step(ep_loss)
            if True:
                print('| EPOCH NR: ', epoch),
                print('| EPOCH LOSS: ', ep_loss)
                print("| AP: ", ap)
                self.writer.add_scalar('ep_det_loss', ep_det_loss, epoch)
                self.writer.add_scalar('ep_iou_loss', ep_iou_loss, epoch)
                self.writer.add_scalar('ep_union_iou_loss', ep_union_iou_loss, epoch)

            print('| DET LOSS: ', ep_det_loss)
            print('| TV LOSS: ', ep_tv_loss)
            print('| IOU LOSS:', ep_iou_loss)
            print('| UNIOU LOSS', ep_union_iou_loss)
            # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            del adv_batch_t, max_prob, det_loss, p_img_batch, tv_loss, loss, adv_batch_mask_t, iou_loss, union_iou_loss
            torch.cuda.empty_cache()

    def generate_patch(self, load_from_file=None, is_random=False):
        # load a image from local patch
        if load_from_file is not None:
            patch = Image.open(load_from_file)
            patch = patch.resize((self.config.patch_size, self.config.patch_size))
            patch = transforms.PILToTensor()(patch) / 255.
            return patch
        if is_random:
            return torch.rand((3, self.config.patch_size, self.config.patch_size))
        return torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)


if __name__ == '__main__':
    pass
