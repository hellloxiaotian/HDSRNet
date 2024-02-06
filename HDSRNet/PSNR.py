import numpy
import numpy as np
import math
import cv2
import torch
import pytorch_ssim
from torch.autograd import Variable

original = cv2.imread("0626.png")  # numpy.adarray
contrast = cv2.imread("KernelGAN_ZSSR_0626.png", 1)


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


psnrValue = psnr(original, contrast)
ssimValue = ssim(original, contrast)
print(psnrValue)
print(ssimValue)
