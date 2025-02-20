import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes, onehot=True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.onehot = onehot

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if self.onehot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                             keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

class LeastSquareGAN(object):
    def __init__(self):
        super(LeastSquareGAN, self).__init__()
        self.real_label = 1.0
        self.fake_label = -1.0

    @staticmethod
    def generator_loss(disc_pred_fake, real_label=1.0):
        loss = 0.5 * torch.mean((disc_pred_fake - real_label) ** 2)
        return loss

    @staticmethod
    def discriminator_loss(disc_pred_real, disc_pred_fake, real_label=1.0, fake_label=-1.0):
        loss = (0.5 * torch.mean((disc_pred_real - real_label) ** 2) +
                0.5 * torch.mean((disc_pred_fake - fake_label) ** 2))
        return loss

    @staticmethod
    def discriminator_fake_loss(disc_pred_fake, fake_label=-1.0):
        loss = 0.5 * torch.mean((disc_pred_fake - fake_label) ** 2)
        return loss

    @staticmethod
    def discriminator_real_loss(disc_pred_real, real_label=1.0):
        loss = 0.5 * torch.mean((disc_pred_real - real_label) ** 2)
        return loss


class VanillaGAN(object):
    def __init__(self):
        super(VanillaGAN, self).__init__()

    @staticmethod
    def generator_loss(disc_pred_fake):
        labels = torch.ones_like(disc_pred_fake)
        loss = F.binary_cross_entropy_with_logits(disc_pred_fake, labels)
        return loss

    @staticmethod
    def discriminator_loss(disc_pred_real, disc_pred_fake):
        real_labels = torch.ones_like(disc_pred_real)
        loss_real = F.binary_cross_entropy_with_logits(disc_pred_real, real_labels)
        fake_labels = torch.zeros_like(disc_pred_fake)
        loss_fake = F.binary_cross_entropy_with_logits(disc_pred_fake, fake_labels)
        loss = loss_real + loss_fake
        return loss

    @staticmethod
    def discriminator_fake_loss(disc_pred_fake):
        labels = torch.ones_like(disc_pred_fake)
        loss = F.binary_cross_entropy_with_logits(disc_pred_fake, labels)
        return loss

    @staticmethod
    def discriminator_real_loss(disc_pred_real):
        labels = torch.ones_like(disc_pred_real)
        loss = F.binary_cross_entropy_with_logits(disc_pred_real, labels)
        return loss

def gradient_penalty(discriminator, x_real, x_fake, gp_weight=10.0):

    if x_fake is None:
        # For real samples, calculate gradients
        x_interpolated = x_real.clone().detach().requires_grad_(True)
    else:
        # For interpolated samples, calculate gradients
        epsilon = torch.rand(x_real.shape[0], 1, 1, 1).to(x_real.device)
        x_interpolated = epsilon * x_real + (1 - epsilon) * x_fake
        x_interpolated.requires_grad_(True)
    # Get discriminator predictions on interpolated samples
    disc_pred_interpolated = discriminator(x_interpolated)

    # Calculate gradients of disc_pred_interpolated with respect to x_interpolated
    gradients = torch.autograd.grad(outputs=disc_pred_interpolated,
                                    inputs=x_interpolated,
                                    grad_outputs=torch.ones_like(disc_pred_interpolated),
                                    create_graph=True,
                                    retain_graph=True)[0]

    # Calculate gradient penalty
    slopes = torch.sqrt(torch.sum(gradients.pow(2), dim=[1, 2, 3]) + 1e-8)
    penalty = torch.mean((slopes - 1.) ** 2)

    # Apply penalty weight
    gp = gp_weight * penalty

    return gp
