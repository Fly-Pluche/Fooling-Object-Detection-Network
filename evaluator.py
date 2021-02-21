import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms


class MaxProbExtractor(nn.Module):
    """
    get the max score in a batch of images
    """

    def __init__(self):
        super(MaxProbExtractor, self).__init__()

    def forward(self, model, batch_image):
        images = torch.unbind(batch_image, 0)
        max_prob_t = torch.cuda.FloatTensor(batch_image.size(0)).fill_(0)
        for i, image in enumerate(images):
            output = model(image)["instances"]
            pred_classes = output.pred_classes
            scores = output.scores
            people_scores = scores[pred_classes == 0]  # select people predict score
            if len(people_scores) != 0:
                max_prob = torch.max(people_scores)
                max_prob_t[i] = max_prob
        return max_prob_t


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


if __name__ == '__main__':
    pass
