import torch 
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

"""
This file contains utility functions that can be used in the main code.
"""

################# Metrics #################
# Assuming images are torch tensors with values in range [0, 1]
def mse_loss(original, enhanced):
    """
    MSE measures the average of the squares of the errors between the original and the enhanced images.

    Value should be close to 0.
    
    Explanation: The lower the MSE, the fewer the differences between the original and the enhanced image. An MSE of 0 indicates perfect similarity, meaning there are no differences at all.
    """

    return F.mse_loss(enhanced, original)

def psnr(original, enhanced):
    """
    PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.
    It is commonly used to measure the quality of reconstruction in image compression.

    Value: Higher values, ideally above 30 dB for high-quality images.
    
    Explanation: PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise. A higher PSNR indicates that the image is of higher quality and closer to the original. Typical values for PSNR in image processing range between 20 and 40 dB, where higher is better. Values above 30 dB typically indicate good quality.
    """

    mse = mse_loss(original, enhanced)
    return 10 * torch.log10(1 / mse)

def ssim(original, enhanced,win_size=7):
    """
    SSIM is a perceptual metric that quantifies the image quality degradation caused by processing such as data compression or by losses in data transmission.
    It considers changes in structural information, luminance, and contrast.
    SSIM ranges from -1 to 1, where 1 indicates perfect similarity.

    Value: Close to 1.
    
    Explanation: SSIM ranges from -1 to 1, where 1 indicates perfect similarity. A value close to 1 means that the structure, luminance, and contrast of the enhanced image are very similar to the original image.
    """
    
    original_np = original.cpu().numpy().transpose(1, 2, 0)
    enhanced_np = enhanced.cpu().numpy().transpose(1, 2, 0)
    return structural_similarity(original_np, enhanced_np, multichannel=True, data_range=1, win_size=win_size, channel_axis=2)




################# a main func to test any func #################
if __name__ == "__main__":
    original_image = torch.rand((3, 256, 256))  # Dummy original image
    enhanced_image = torch.rand((3, 256, 256))  # Dummy enhanced image

    mse = mse_loss(original_image, enhanced_image)
    psnr_value = psnr(original_image, enhanced_image)
    ssim_value = ssim(original_image, enhanced_image)

    print(f"MSE: {mse.item()}")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")