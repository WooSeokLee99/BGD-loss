import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

#####################################################
# MEMO #############################################
#####################################################

"""
Boundary Gaussian Difference Loss

Paper : Boundary Gaussian Distance Loss Function for Enhancing Character Extraction
        from High-Resolution Scans of Ancient Metal-Type Printed Books
DOI   : https://www.mdpi.com/2079-9292/13/10/1957

 BGD loss allows the segmentation network to accurately extract character strokes that can be
attributed to the typeface of the movable metal type used for printing. Our method calculates
deviation between the boundary of predicted character strokes and the counterpart of
the ground-truth strokes.

 Diverging from traditional Euclidean distance metrics, our approach determines the deviation
indirectly utilizing boundary pixel-value difference over a Gaussian-smoothed version of the
stroke boundary. This approach helps extract characters with smooth boundaries efficiently.

 Through experiments, it is confirmed that the proposed method not only smoothens stroke boundaries 
in character extraction, but also effectively eliminates noise and outliers, significantly improving
the clarity and accuracy of the segmentation process.
"""

#####################################################
# Loss Function #####################################
#####################################################


def BoundaryGaussianDifference_loss(logits, targets, channel_dim: int = 1, sigma: int =6.0, eps: float = 1e-6):
    # 
    # https://www.mdpi.com/2079-9292/13/10/1957
    
    # Binarization

    logits  = binarization(logits, dim=channel_dim) # For erosion, logits must be a binary image!
    targets = targets.float()

    # Boundary: B(Y) = Y - E(Y)
    #           V = (x,y)|B(x,y) > 0

    B_pr = logits  - torch_erosion_3D(logits.unsqueeze(1))
    B_gt = targets - torch_erosion_3D(targets.unsqueeze(1))
    
    # Gaussian Blur: G = 2*pi*σ^2 * exp(-(x^2+y^2)/(2*σ^2))

    device = logits.device
    kernel_size = (sigma * 4 + 1, sigma * 4 + 1)
    normalization = sigma * torch.sqrt(torch.tensor(2) * torch.pi).to(device)

    G_pr = transforms.functional.gaussian_blur(B_pr, kernel_size, [sigma, sigma]) * normalization
    G_gt = transforms.functional.gaussian_blur(B_gt, kernel_size, [sigma, sigma]) * normalization

    # Calc Boundary Distanve Value: M(G,V) = (1/V) * (Σ_{(i,j)∈V} G(i,j))

    B_pr_sum = B_pr.sum() + eps
    B_gt_sum = B_gt.sum() + eps

    GpVp = (G_pr * B_pr).sum()
    GpVg = (G_pr * B_gt).sum()
    GgVg = (G_gt * B_gt).sum()
    GgVp = (G_gt * B_pr).sum()

    # Calc Boundary Gaussian Differnce: Loss = [M(G,V)-M(G,V')]+[M(G',V')-M(G'-V)]
    #  let notation ' is matrix from GT

    # BGD = M(G,V) - M(G,V') + M(G',V') - M(G'-V) 
    #     = M(G,V) - M(G'-V) + M(G',V') - M(G,V')
    #     = GpVp / B_pr_sum  - GgVp / B_pr_sum + GgVg / B_gt_sum - GpVg / B_gt_sum
    #     = (GpVp - GgVp) / B_pr_sum + (GgVg - GpVg) / B_gt_sum

    return (GpVp - GgVp) / B_pr_sum + (GgVg - GpVg) / B_gt_sum

def BoundaryGaussianDifference_loss_v2(input, target, channel_dim: int = 1, sigma: int =6.0, eps: float = 1e-6): # Paper version

    # Binarization

    logits  = binarization(logits, dim=channel_dim) # For erosion, logits must be a binary image!
    targets = targets.float()

    # Boundary

    B_pr = input  - torch_erosion_3D(input, 1)
    B_gt = target - torch_erosion_3D(target, 1)
    
    # Gaussian Blur

    device = logits.device
    kernel_size = (sigma * 4 + 1, sigma * 4 + 1)
    normalization = sigma * torch.sqrt(torch.tensor(2) * torch.pi).to(device)
                                                                      
    G_pr = transforms.functional.gaussian_blur(B_pr, kernel_size, sigma=(sigma, sigma)) * normalization
    G_gt = transforms.functional.gaussian_blur(B_gt, kernel_size, sigma=(sigma, sigma)) * normalization

    # Calc Boundary Gaussian Differnce

    B_pr_sum = B_pr.sum() + eps
    B_gt_sum = B_gt.sum() + eps

    D = G_pr - G_gt

    return (D * B_pr).sum() / B_pr_sum + (-D * B_gt).sum() / B_gt_sum

#####################################################
# Util functions ####################################
#####################################################

def binarization(img, dim=1):

    prob = torch.softmax(img, dim)

    img_ = torch.argmax(prob, dim).float()
    #img_ = torch.sigmoid(prob[:,dim,:,:] * 50) #recommanded
    #img_ = softargmax(img, dim)

    return img_
    
def softargmax(logits, dim=1, beta: float =10.0): # v5
    # https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
    
    probs = torch.softmax(logits * beta, dim=dim)  # [B,C,H,W]
    idx = torch.arange(logits.size(dim), device=logits.device, dtype=probs.dtype).view(1, -1, 1, 1)  # [0,1]
    return (probs * idx).sum(dim=dim)

def torch_erosion_3D(img, kernel_size=3):
    """
    from hausdorff_distance_loss.py
    """
    
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device, dtype=img.dtype)
    img = F.conv2d(img, kernel, padding=1, bias=None)
    img = (img >= (kernel_size*kernel_size)).to(img.dtype)

    return img

@torch.no_grad()
def show_img(x):

    """
    Function for plotting
    Input: [H,W]
    """

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    plt.imshow((x > 0).astype(np.float32), cmap="gray", vmin=0, vmax=1)
    plt.axis("off"); plt.show()