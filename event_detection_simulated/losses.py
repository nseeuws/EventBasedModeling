import torch
from torch import Tensor
import torch.nn.functional as F 
from torch import nn 


def stable_softplus_torch(x: Tensor) -> Tensor:
    return F.softplus(-torch.abs(x)) + F.relu(x)


def iou_loss(dur_target: Tensor, dur_pred: Tensor) -> Tensor:
    mask = torch.ne(dur_target, 0)

    target = torch.masked_select(dur_target, mask=mask)
    prediction = torch.masked_select(dur_pred, mask=mask)

    iou = torch.minimum(target, prediction) / torch.maximum(target, prediction)

    return torch.sum(1. - iou)


def focal_loss_functional(
    center_target: Tensor, center_pred: Tensor, logit_pred: Tensor,
    alpha: Tensor, beta: Tensor, a_t: Tensor
) -> Tensor:
    # Background term (target != 1)
    background_term = torch.multiply(
        torch.pow(1. - center_target, beta),
        torch.pow(center_pred, alpha)
    )
    background_term = torch.multiply(
        background_term, -stable_softplus_torch(logit_pred)
    )
    background_term = a_t * torch.sum(background_term)

    # Center term (target == 1)
    mask = torch.eq(center_target, 1.)
    location = torch.masked_select(center_pred, mask=mask)
    logit = torch.masked_select(logit_pred, mask=mask)
    center_term = torch.multiply(
        torch.pow(1. - location, alpha),
        logit - stable_softplus_torch(logit)
    )
    center_term = (1 - a_t) * torch.sum(center_term)

    return - (background_term + center_term)


class FocalLoss(nn.Module):
    """This things is stateful, move to GPU!

    """
    def __init__(self, alpha: float, beta: float, a_t: float, device) -> None:
        super().__init__()
        assert alpha > 0.
        assert beta > 0.
        assert 0. <= a_t <= 1.

        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)
        self.a_t = torch.tensor(a_t, dtype=torch.float32, device=device)

    def forward(
        self, center_target: Tensor, center_pred: Tensor,
        logit_pred: Tensor
    ) -> Tensor:
        return focal_loss_functional(
            center_target=center_target, center_pred=center_pred,
            logit_pred=logit_pred, alpha=self.alpha,
            beta=self.beta, a_t=self.a_t
        )