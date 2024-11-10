import torch
import torch.nn.functional as F

from guided_filter import ColorGuidedFilter


def get_A(I_dc, I, top_p=0.1):
    B, C, H, W = I.shape

    q = 1 - top_p / 100
    threshold = torch.quantile(I_dc.view(B, -1), q, dim=1)  # (B,)

    mask = I_dc >= threshold.view(B, 1, 1, 1)  # (B, 1, H, W)

    masked_I = I * mask  # (B, C, H, W)
    sum_A = masked_I.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    counts = mask.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

    return sum_A / counts


def get_dark_channel(I, patch_size):
    return -F.max_pool2d(
        -torch.min(I, dim=1, keepdim=True).values,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )


def dcp_dehaze_pt(I, patch_size=15):
    """
    I : (B, C, H, W)
    patch_size : int
    """
    I_dc = get_dark_channel(I, patch_size)  # (B, 1, H, W)

    A = get_A(I_dc, I)  # (B, C, 1, 1)

    t = 1 - 0.9 * get_dark_channel(I / A, patch_size)  # (B, 1, H, W)

    t_ref = ColorGuidedFilter(30)(I, t)  # (B, 1, H, W)

    J = (I - A) / t_ref + A  # (B, C, H, W)
    J = J.clamp(0, 1)

    return J, t_ref
