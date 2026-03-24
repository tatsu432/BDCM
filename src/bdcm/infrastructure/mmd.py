from __future__ import annotations

import torch


def mmd(
    x: torch.Tensor, y: torch.Tensor, kernel: str, device: torch.device
) -> torch.Tensor:
    """Empirical maximum mean discrepancy (same kernels as original)."""
    x = x.to(device)
    y = y.to(device)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX = torch.zeros(xx.shape, device=device)
    YY = torch.zeros(xx.shape, device=device)
    XY = torch.zeros(xx.shape, device=device)

    if kernel == "multiscale":
        for a in (0.2, 0.5, 0.9, 1.3):
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        for a in (10, 15, 20, 50):
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)
