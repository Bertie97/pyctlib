#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package micomputing
##############################

import sys
sys.path.append("/Users/admin/Documents/BJ_Files/_Courses_/Research/zycmodules")
sys.path.append("/Users/admin/Documents/BJ_Files/_Courses_/Research/zycmodules/pyctlib")
import torch
import torchplus as tp

######## Section 1: Mutual Information ########
eps = 1e-6

def Bspline(i, U):
    i = tp.tensor(i); U = tp.tensor(U)
    return (
        tp.where(i == -1, (1 - U) ** 3 / 6,
        tp.where(i == 0, U ** 3 / 2 - U * U + 2 / 3,
        tp.where(i == 1, (- 3 * U ** 3 + 3 * U * U + 3 * U + 1) / 6,
        tp.where(i == 2, U ** 3 / 6,
        tp.zeros_like(U)))))
    )

def dBspline(i, U):
    i = tp.tensor(i); U = tp.tensor(U)
    return (
        tp.where(i == -1, - 3 * (1 - U) ** 2 / 6,
        tp.where(i == 0, 3 * U ** 2 / 2 - 2 * U,
        tp.where(i == 1, (- 3 * U ** 2 + 2 * U + 1) / 2,
        tp.where(i == 2, 3 * U ** 2 / 6,
        tp.zeros_like(U)))))
    )

def dBspline_WRT_I1(i, U):
    '''
    THe derivative of Bspline function with respect to I2.
    i, U: n_batch x n_hist x n_data
    '''
    return dBspline(i[:, 0], U[:, 0]) * Bspline(i[:, 1], U[:, 1])

def dBspline_WRT_I2(i, U):
    '''
    THe derivative of Bspline function with respect to I2.
    i, U: n_batch x n_hist x n_data
    '''
    return Bspline(i[:, 0], U[:, 0]) * dBspline(i[:, 1], U[:, 1])

class JointHistogram(tp.autograd.Function):

    @staticmethod
    def forward(ctx, I1, I2, nbin=100):
        with tp.no_grad():
            if hasattr(ctx, 'JH'): del ctx.JH
            nbin = tp.tensor(nbin)
            data_pair = tp.stack(I1.flatten(1), I2.flatten(1), dim={1})
            nbatch, nhist, ndata = data_pair.ishape
            indices = []; values = []
            ctx.window = (tp.image_grid(4, 4) - 1).flatten(1).transpose(0, 1)
            for shift in ctx.window:
                # [nbatch] x {nhist} x ndata
                hist_pos = data_pair * nbin
                index = tp.clamp(tp.floor(hist_pos).long() + shift, 0, nbin - 1)
                batch_idx = tp.arange(nbatch).expand_to([nbatch], {1}, ndata)
                index = tp.cat(batch_idx, index, 1)
                value = Bspline(shift.expand_to(data_pair), tp.decimal(hist_pos)).prod(1)
                indices.append(index)
                values.append(value)
            # n_batch x (1 + n_hist) x (n_data x 4 ** n_hist)
            Mindices = tp.cat(indices, -1)
            # n_batch x (n_data x 4 ** n_hist)
            Mvalues = tp.cat(values, -1)
            # (1 + n_hist) x (n_batch x n_data x 4 ** n_hist)
            indices = Mindices.transpose(0, 1).flatten(1)
            # (n_batch x n_data x 4 ** n_hist)
            values = Mvalues.flatten(0)
            if tp.Device == tp.DeviceCPU: creator = torch.sparse.FloatTensor
            else: creator = torch.cuda.sparse.FloatTensor
            collected = creator(indices, values, (nbatch, nbin, nbin)).to_dense()
            collected = tp.Tensor(collected, batch_dim=0)

            ctx.nbin = nbin
            ctx.Ishape = I1.shape
            ctx.data_pair = data_pair
            ctx.JH = collected / ndata
        return ctx.JH

    @staticmethod
    def backward(ctx, grad_output):
        with tp.no_grad():
            nbin = ctx.nbin
            data_pair = ctx.data_pair
            nbatch, nhist, ndata = data_pair.ishape
            dPdI1 = torch.zeros(ctx.Ishape)
            dPdI2 = torch.zeros(ctx.Ishape)
            for shift in ctx.window:
                # [nbatch] x {nhist} x ndata
                shift = shift.view(1, 2, 1)
                hist_pos = data_pair * nbin
                index = torch.clamp(torch.floor(hist_pos).long() + shift, 0, nbin - 1)
                grad_y = grad_output[(slice(None),) + index.split(1, 1)].squeeze(2)
                value = grad_y.gather(0, torch.arange(nbatch).long().unsqueeze(0).unsqueeze(-1).repeat(1, 1, ndata)).view(ctx.Ishape)
                dPdI1 += value * dBspline_WRT_I1(shift, tp.decimal(data_pair * nbin)).view(ctx.Ishape)
                dPdI2 += value * dBspline_WRT_I2(shift, tp.decimal(data_pair * nbin)).view(ctx.Ishape)
        return dPdI1, dPdI2, None

def MutualInformation(A, B, nbin=100):
    assert A.has_batch and B.has_batch
    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - tp.sum(Pab * tp.log2(tp.where(Pab < eps, tp.ones_like(Pab), Pab)), [1, 2])
    Hx = - tp.sum(Pa * tp.log2(tp.where(Pa < eps, tp.ones_like(Pa), Pa)), 1)
    Hy = - tp.sum(Pb * tp.log2(tp.where(Pb < eps, tp.ones_like(Pb), Pb)), 1)
    return Hx + Hy - Hxy

def NormalizedMutualInformation(A, B, nbin=100):
    assert A.has_batch and B.has_batch
    Pab = JointHistogram.apply(A, B, nbin)
    Pa = Pab.sum(2); Pb = Pab.sum(1)
    Hxy = - tp.sum(Pab * tp.log2(tp.where(Pab < eps, tp.ones_like(Pab), Pab)), [1, 2])
    Hx = - tp.sum(Pa * tp.log2(tp.where(Pa < eps, tp.ones_like(Pa), Pa)), 1)
    Hy = - tp.sum(Pb * tp.log2(tp.where(Pb < eps, tp.ones_like(Pb), Pb)), 1)
    return (Hx + Hy) / Hxy

###############################################

######## Section 2: Cross Correlation #########

def local_matrix(A, B, s=0, kernel="Gaussian", kernel_size=3):
    if isinstance(kernel, str):
        if kernel.lower() == "gaussian": kernel = tp.gaussian_kernel(n_dims = A.nspace, kernel_size = kernel_size).unsqueeze(0, 0)
        elif kernel.lower() == "mean": kernel = tp.ones(*(kernel_size,) * A.nspace).unsqueeze(0, 0)
    elif hasattr(kernel, 'shape'): kernel_size = kernel.size(-1)
    mean = lambda a: eval("tp.nn.functional.conv%dd"%A.nspace)(a.unsqueeze(), kernel, padding = kernel_size // 2).squeeze(0)

    if s > 0:
        GA = tp.grad_image(A)
        GB = tp.grad_image(B)
        point_estim = tp.stack(tp.dot(GA, GA), tp.dot(GA, GB), tp.dot(GB, GB), dim={1})
    else: point_estim = 0

    RA = A - mean(A)
    RB = B - mean(B)
    local_estim = tp.stack(RA * RA, RA * RB, RB * RB, dim={1})

    return s * point_estim + local_estim

def LocalCrossCorrelation(A, B, s=0, kernel="Gaussian", kernel_size=3):
    assert A.has_batch and B.has_batch
    S11, S12, S22 = local_matrix(A, B, s=0, kernel="Gaussian", kernel_size=3).split(1, 1)
    num = S12.abs().squeeze(1)
    den = tp.sqrt(S11 * S22).squeeze(1)
    return tp.where(den.abs() < 1e-6, tp.zeros_like(num), num / den.clamp(min=1e-6)).mean()

###############################################

########## Section 3: Local Gradient ##########

def NormalizedVectorInformation(A, B):
    assert A.has_batch and B.has_batch and A.has_channel and B.has_channel
    GA = tp.grad_image(A)
    GB = tp.grad_image(B)
    return (tp.dot(GA, GB) / tp.sqrt(tp.dot(GA, GB) * tp.dot(GB, GB))).mean()

###############################################

####### Section 4: Intensity Difference #######

def SumSquaredDifference(A, B):
    assert A.has_batch and B.has_batch
    A.remove_channel()
    B.remove_channel()
    return ((A - B) ** 2).sum()

def MeanSquaredErrors(A, B):
    assert A.has_batch and B.has_batch
    A.remove_channel()
    B.remove_channel()
    return ((A - B) ** 2).mean()

###############################################

##### Section 5: Distribution Similarity ######

def CrossEntropy(y, label):
    assert y.has_batch and label.has_batch and y.has_channel and label.has_channel
    ce = - label * tp.log(y.clamp(1e-10, 1.0))
    return ce.sum(ce.channel_dimension).mean()

def CrossCorrelation(A, B):
    assert A.has_batch and B.has_batch
    dA = A - A.mean(); dB = B - B.mean()
    return (dA * dB).sum()

def NormalizedCrossCorrelation(A, B):
    assert A.has_batch and B.has_batch
    dA = A - A.mean(); dB = B - B.mean()
    return (dA * dB).sum() / (dA ** 2).sum().sqrt() / (dB ** 2).sum().sqrt()

###############################################

########## Section 6: Region Overlap ##########

def Dice(A, B, multi_label = False):
    '''
    if multi_label:
        A: (n_batch, n_label, n_1, n_2, ..., n_k)
        B: (n_batch, n_label, n_1, n_2, ..., n_k)
        return: (n_batch, n_label)
    else:
        A: (n_batch, n_1, n_2, ..., n_k)
        B: (n_batch, n_1, n_2, ..., n_k)
        return: (n_batch,)
    '''
    assert A.has_batch and B.has_batch
    ABsum = A.sum() + B.sum()
    return 2 * (A * B).sum() / (ABsum + eps)

def LabelDice(A, B, class_labels=None):
    '''
    :param A: (n_batch, n_1, ..., n_k)
    :param B: (n_batch, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    assert A.has_batch and B.has_batch
    if not class_labels: class_labels = sorted(I1.unique().tolist())
    A_labels = [1 - tp.clamp(tp.abs(A - i), 0, 1) for i in class_labels]
    B_labels = [1 - tp.clamp(tp.abs(B - i), 0, 1) for i in class_labels]
    A_maps = tp.stack(A_labels, {1})
    B_maps = tp.stack(B_labels, {1})
    return Dice(A_maps, B_maps)

###############################################
