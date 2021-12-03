import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


def produce_cycle_mask(img_h, img_w, cycle_size=8):
    """
    produce a cycle mask
    Args:
        img_h: the height of the image
        img_w: the width of the image
        cycle_size: use to control the size of the cycle
    """
    lpf = torch.zeros((img_h, img_w))
    R = (img_h + img_w) // cycle_size
    for x in range(img_w):
        for y in range(img_h):
            if ((x - (img_w - 1) / 2) ** 2 + (y - (img_h - 1) / 2) ** 2) < (R ** 2):
                lpf[y, x] = 1
    hpf = 1 - lpf
    return lpf, hpf


def pytorch_fft(img, lpf, hpf):
    """
    use torch.fft to produce two images (the image only with low frequency and the image only with high frequency)
    Args:
        img: the img you want to process
        lpf: the low frequency mask
        hpf: the high frequency mask
    """
    f = fft.fftn(img.unsqueeze(0), dim=(2, 3))
    f = torch.roll(f, (img.size(1) // 2, img.size(2) // 2), dims=(2, 3))
    f_l = f * lpf
    f_h = f * hpf
    img_low_frequency = torch.abs(fft.ifftn(f_l, dim=(2, 3)))
    img_high_frequency = torch.abs(fft.ifftn(f_h, dim=(2, 3)))
    return img_low_frequency, img_high_frequency


"""
Y=0.299R+0.587G+0.114B
Cb=0.564(B-Y)
Cr=0.713(R-Y)
"""


def rgb2ycbcr(img):
    img_ycbcr = img.clone()
    img_ycbcr[0, :, :] = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    img_ycbcr[1, :, :] = 0.564 * (img[2, :, :] - img_ycbcr[0, :, :])
    img_ycbcr[2, :, :] = 0.713 * (img[0, :, :] - img_ycbcr[0, :, :])
    return img_ycbcr


def ycbcr2rgb(img):
    img_rgb = img.clone()
    # B
    img_rgb[2, :, :] = (img[1, :, :] / 0.564) + img[0, :, :]
    # R
    img_rgb[0, :, :] = (img[2, :, :] / 0.713) + img[0, :, :]
    # G
    img_rgb[1, :, :] = (img[0, :, :] - 0.299 * img_rgb[0, :, :] - 0.114 * img_rgb[2, :, :]) / 0.587
    return img_rgb


def mask_fft(img, alpha):
    """
    Use dynamic mask to calculate the corresponding fft weight.
    :param img:the img_patch you want to process
    :param alpha: random tensor used to optimization to select the low and high frequency with better attack.
    alpha:3xHxW
    :return:a weighted img_path
    """
    f = fft.fftn(img.unsqueeze(0), dim=(2, 3))
    f = torch.roll(f, (img.size(1) // 2, img.size(2) // 2), dims=(2, 3))
    f = alpha * f
    img_frequency = torch.abs(fft.ifftn(f, dim=(2, 3)))

    return img_frequency


def mask_fft2(img, alpha):
    """
    Use dynamic mask to calculate the corresponding fft weight.
    :param img:the img_patch you want to process
    :param alpha: random tensor used to optimization to select the low and high frequency with better attack.
    alpha:3xHxW
    :return:a weighted img_path
    """
    img = rgb2ycbcr(img)
    f = fft.fftn(img.unsqueeze(0), dim=(2, 3))
    f = torch.roll(f, (img.size(1) // 2, img.size(2) // 2), dims=(2, 3))
    f = alpha * f
    img_frequency = torch.abs(fft.ifftn(f, dim=(2, 3)))
    img_frequency = ycbcr2rgb(img_frequency.squeeze(0)).unsqueeze(0)
    return img_frequency


class FrequencyAttention(nn.Module):
    def __init__(self):
        super(FrequencyAttention, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 3, 1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.attention = None
        # self.bn = torch.nn.BatchNorm2d(num_features=3)
        # self.frequency_mask = torch.ones(3, 950, 950)

    def forward(self, x, ycbcr=False):
        if ycbcr:
            x = rgb2ycbcr(x)
        f = fft.fftn(x.unsqueeze(0), dim=(2, 3))
        f = torch.roll(f, (x.size(1) // 2, x.size(2) // 2), dims=(2, 3))
        frequency_attention = self.conv1(f.to(torch.float32))
        frequency_attention = self.sigmoid1(frequency_attention)
        frequency_attention = self.conv2(frequency_attention)
        frequency_attention = self.sigmoid2(frequency_attention)
        self.attention = frequency_attention.clone()
        f = frequency_attention * f
        x = torch.abs(fft.ifftn(f, dim=(2, 3)))
        if ycbcr:
            x = ycbcr2rgb(x.squeeze(0))
        else:
            x = x.squeeze(0)
        return x
