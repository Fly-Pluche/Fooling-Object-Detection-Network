import torch
import torch.fft as fft




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
