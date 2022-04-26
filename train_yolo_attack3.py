"""
论文补充实验
yolov4 补充实验
"""

from __future__ import absolute_import

import warnings

import torch.cuda
from evaluator import *
from evaluator import PatchEvaluator
from patch import *
from patch_config import *
from torch.utils.tensorboard import SummaryWriter
from utils.frequency_tools import *
from utils.transforms import CMYK2RGB
from utils.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
        self.model_ = Yolov4(self.config.model_path)
        self.name = '八角 mask_FFT_隐式训练-18step-lr=0.005 ycbcr YOLOv4[finial-pre-5]'
        self.log_path = self.config.log_path
        self.writer = self.init_tensorboard(name='base')
        self.init_logger()
        os.mkdir(os.path.join(self.log_path, 'visual_image'))
        self.image_save_path = os.path.join(self.log_path, 'visual_image')
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()
        self.max_extractor = MaxProbExtractor().cuda()
        self.total_variation = TotalVariation().cuda()
        self.union_detector = UnionDetectorBCE().cuda()
        self.frequency_loss = FrequencyLoss(self.config).cuda()
        self.ms_ssim_loss = MS_SSIMLOSS().cuda()
        self.patch_evaluator = None
        self.entropy = nn.BCELoss()
        self.is_cmyk = self.config.is_cmyk

    def init_logger(self):
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        detail_info = self.config.detail_info
        filename = os.path.join(self.log_path, 'log.log')
        logging.basicConfig(filename=filename, level=logging.DEBUG, format=LOG_FORMAT)
        logging.info(f'start learning rate: {self.config.start_learning_rate}')
        logging.info(f'step_size: {self.config.step_size}')
        logging.info(f'gamma: {self.config.gamma}')
        logging.info(f'patch size: {self.config.patch_size}')
        logging.info(f'batch size: {self.config.batch_size}')
        logging.info(f'img size: {self.config.img_size}')
        logging.info(f'fft_size:{self.config.fft_size}')
        logging.info(f'model image size: {self.config.model_image_size}')
        logging.info(f'cmyk: : {self.config.is_cmyk}')
        logging.info(f'optimizer: {self.config.optim}')
        logging.info(f'patch scale: {self.config.patch_scale}')
        logging.info(f'detail: {detail_info}')

    def init_tensorboard(self, name=None):
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.log_path = f'{self.config.log_path}/{time_str}_{name}_{self.name}'
            return SummaryWriter(f'{self.config.log_path}/{time_str}_{name}_{self.name}')
        else:
            return SummaryWriter()

    def train(self, load_patch_from_file=None, load_mask_from_file=None, is_random=True):
        """
         optimizer a adversarial patch
        """
        # load train datasets
        datasets = ListDataset(self.config.coco_train_txt)
        train_data = DataLoader(
            datasets,
            batch_size=self.config.batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True
        )
        test_data = DataLoader(
            ListDataset(self.config.coco_val_txt),
            num_workers=16,
            batch_size=self.config.batch_size,
            drop_last=True
        )
        self.patch_evaluator = PatchEvaluator(self.model_, test_data, use_deformation=False).cuda()
        self.asr_calculate = ObjectVanishingASR(self.config.img_size, use_deformation=False)
        self.asr_calculate.register_dataset(self.model_, test_data)
        epoch_length = len(train_data)

        # generate a rgb patch
        adv_patch_cpu = self.generate_patch(load_patch_from_file, is_random=True, is_cmyk=self.is_cmyk)
        adv_patch_cpu.requires_grad_(True)

        adv_mask_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 1.0)
        adv_mask_cpu.requires_grad_(True)

        adv_patch = None
        adv_mask = None

        if self.config.optim == 'adam':
            optimizer1 = torch.optim.Adam([{"params": adv_patch_cpu, 'lr': self.config.start_learning_rate},
                                           {"params": adv_mask_cpu, 'lr': 0.005}])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[18, 36, 56, 70],
                                                             gamma=self.config.gamma,
                                                             last_epoch=-1)  # used to update learning rate

        elif self.config.optim == 'sgd':
            optimizer1 = optim.SGD([adv_patch_cpu], momentum=0.9, lr=self.config.start_learning_rate)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[10, 25, 60, 100, 190], gamma=0.5,
                                                       last_epoch=-1)

        else:
            raise ValueError("Optimizer can only be adam or sgd!")

        # start training
        min_ap = 1
        max_asr = 0
        for epoch in range(10000):
            ep_iou_all = 0
            ep_det_loss = 0
            ep_conf_union_loss = 0
            ep_union_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            i_batch = 0
            for image_batch, people_boxes, labels_batch in tqdm(train_data):
                i_batch += 1
                print(self.log_path)
                image_batch = image_batch.cuda()
                labels_batch = labels_batch.cuda()
                people_boxes = people_boxes.cuda()
                adv_patch = adv_patch_cpu.cuda()
                adv_mask = adv_mask_cpu.cuda()
                # 隐式训练 rgb
                # adv_patch = mask_fft(adv_patch, adv_mask).squeeze(0)
                # 隐式训练 ycbcr -》 前面的阶段使用frequency attention进行引导
                if epoch < 20:
                    adv_patch = mask_fft2(adv_patch, adv_mask).squeeze(0)
                if self.is_cmyk:
                    adv_patch = CMYK2RGB(adv_patch)

                # adv_patch = torch.clamp(adv_patch, 0, 0.999999)
                # Attach the attack image to the clothing
                adv_batch_t = self.patch_transformer(adv_patch, people_boxes, labels_batch.clone())
                p_img_batch = self.patch_applier(image_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (self.config.img_size[1], self.config.img_size[0]),
                                            mode='bilinear')

                det_loss = torch.mean(self.max_extractor(self.model_, p_img_batch))
                tv_loss = self.total_variation(adv_patch)

                # calculate each part of the lossle
                logging.info(
                    f'epoch: {epoch} iter: {i_batch} |det loss: {det_loss},tv loss:{tv_loss}')

                loss1 = det_loss + 2.5 * tv_loss
                loss = loss1

                # evaluate
                ep_det_loss += det_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                print(loss)
                optimizer1.step()
                optimizer1.zero_grad()

                adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range
                # adv_mask_cpu.data = torch.sigmoid(adv_mask_cpu.data)

                if i_batch % 23 == 0:
                    iteration = epoch_length * epoch + i_batch
                    self.writer.add_scalar('det_loss', det_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('TV_loss', tv_loss.detach().cpu().numpy(), iteration)
                    # self.writer.add_scalar('Frequency_loss', frequency_loss.detach().cpu().numpy(), iteration)
                    # plt.imshow(np.asarray(functional.to_pil_image(adv_patch_cpu)))
                    # plt.show()
                if i_batch % 699 == 0:
                    iteration = epoch_length * epoch + i_batch
                    self.writer.add_image('adv-patch', adv_patch.clone().cpu(), iteration)
                    self.writer.add_image('frequency-attention', adv_mask.clone().cpu(), iteration)

                del adv_batch_t, p_img_batch, det_loss, tv_loss, loss1, loss

            # visual attack effection
            if epoch % 10 == 0:
                self.patch_evaluator.save_visual_images(adv_patch.clone(), self.image_save_path, epoch)

            # eval patch
            with torch.no_grad():
                if self.is_cmyk:
                    adv_cmyk = adv_patch_cpu.clone()
                    adv = adv_cmyk.cuda()
                    adv = CMYK2RGB(adv)
                else:
                    adv = adv_patch.clone()
                    frequency_attention_mask = adv_mask.clone()

                # calculate ap50
                ap = self.patch_evaluator(adv, 0.5)
                self.writer.add_scalar('ap', ap, epoch)
                # calculate asr50
                self.asr_calculate.inference_on_dataset(adv, clean=False)
                ASR, ASRs, ASRm, ASRl = self.asr_calculate.calculate_asr()
                print(self.log_path)
                self.writer.add_scalar('ASR', ASR, epoch)
                self.writer.add_scalar('ASRs', ASRs, epoch)
                self.writer.add_scalar('ASRm', ASRm, epoch)
                self.writer.add_scalar('ASRl', ASRl, epoch)

                # save better patch image
                if float(ap) < min_ap:
                    name = os.path.join(self.log_path, str(float(ap) * 100))
                    torchvision.utils.save_image(adv, name + '.png')
                    # torch2raw(adv.cpu(), name + '.raw')
                    min_ap = float(ap)
                    if self.is_cmyk:
                        name2 = os.path.join(self.log_path, str(float(ap) * 100) + '.TIF')
                        adv_cmyk = functional.to_pil_image(adv_cmyk)
                        adv_cmyk.save(name2)
                if float(ASR) > max_asr:
                    name = os.path.join(self.log_path, str(float(ASR) * 100)[:4] + '_asr')
                    name2 = os.path.join(self.log_path, str(float(ASR) * 100)[:4] + '_asr_mask')
                    torchvision.utils.save_image(adv, name + '.png')
                    torchvision.utils.save_image(frequency_attention_mask, name2 + '.png')
                    # torch2raw(adv.cpu(), name + '.raw')
                    torch.save(adv_mask, name2 + '.pt')
                    torch.save(adv, name + '.pt')
                    max_asr = float(ASR)

            ep_tv_loss = ep_tv_loss / len(train_data)
            ep_det_loss = ep_det_loss / len(train_data)
            ep_loss = ep_loss / len(train_data)
            scheduler.step()

            if True:
                logging.info(f'epoch: {epoch}| EPOCH NR: {epoch}'),
                logging.info(f'epoch: {epoch}| EPOCH LOSS: {ep_loss}')
                logging.info(f"epoch: {epoch}| AP: {ap}")
                logging.info(f"epoch: {epoch}| LR: {optimizer1.param_groups[0]['lr']}")
                self.writer.add_scalar('ep_iou_all', ep_iou_all, epoch)
                self.writer.add_scalar('ep_conf_union_loss', ep_conf_union_loss, epoch)
                self.writer.add_scalar('ep_union_loss', ep_union_loss, epoch)
                self.writer.add_scalar('ep_loss', ep_loss, epoch)
                self.writer.add_scalar('AP', ap, epoch)
                self.writer.add_scalar('LR', optimizer1.param_groups[0]['lr'], epoch)

            # logging.info(f'epoch: {epoch} | ep_iou_all: {ep_iou_all}')
            logging.info(f'epoch: {epoch} | ep_tv_loss: {ep_tv_loss}')
            logging.info(f'epoch: {epoch} | ep_det_loss: {ep_det_loss}')
            # logging.info(f'epoch: {epoch} | ep_conf_union_loss: {ep_conf_union_loss}')
            # logging.info(f'epoch: {epoch} | ep_union_loss: {ep_union_loss}')
            # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            # del max_prob, det_loss, p_img_batch, tv_loss, loss, adv_batch_mask_t, iou_loss, union_iou_loss
            # torch.cuda.empty_cache()

    def generate_patch(self, load_from_file=None, is_random=False, is_cmyk=0, is_from_pt=False):
        # print('load_from_file', load_from_file)

        # load a image from local patch
        if load_from_file is not None:
            if is_from_pt:
                patch = torch.load(load_from_file)
            elif 'raw' in load_from_file:
                patch = raw2torch(load_from_file, np.array([3, self.config.patch_size, self.config.patch_size]))
            else:
                patch = Image.open(load_from_file)
                patch = patch.resize((self.config.patch_size, self.config.patch_size))
                patch = transforms.PILToTensor()(patch) / 255.
            return patch
        elif is_random:
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
