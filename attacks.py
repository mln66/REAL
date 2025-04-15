import torch
import torch.nn as nn
import torch.optim as optim

from math import exp
from functools import partial

from cw_attack import L2Adversary
from df_attack import DeepFool
import math
def empty(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    return torch.zeros_like(X)

def inject_noise(X, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    return (X + torch.randn_like(X) * epsilon).clamp(*bound) - X

def fgsm(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    if y is None:
        loss = criterion(model, X + delta)
    else:
        try:
            criterion_name = criterion.func.__name__
            if criterion_name == 'second_order':
                loss = criterion(model, X + delta, y)
            else:
                loss = criterion(model(X + delta), y)
        except:
            loss = criterion(model(X + delta), y)
    loss.backward()
    if y is None:
        delta = epsilon * delta.grad.detach().sign()
    else:
        delta = epsilon * delta.grad.detach().sign()
    return (X + delta).clamp(*bound) - X




def pgd_linf(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        if y is None:
            # print("t:{}".format(t))
            # print("X+delta.shape:{}".format((X+delta).shape))
            loss = criterion(model, X + delta)
        else:
            try:
                criterion_name = criterion.func.__name__
                if criterion_name == 'second_order':
                    loss = criterion(model, X + delta, y)
                else:
                    loss = criterion(model(X + delta), y)
            except:
                loss = criterion(model(X + delta), y)
        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        delta.grad.zero_()

    delta.data = (X + delta).clamp(*bound) - X
    return delta.detach()

def pgd_linf_1(model, criterion, X, X_denoise,y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X_denoise, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X_denoise, requires_grad=True)

    for t in range(num_iter):
        if y is None:
            # print("t:{}".format(t))
            # print("X+delta.shape:{}".format((X+delta).shape))
            # loss = criterion(model,X + delta,X_denoise + delta)
            loss = criterion(model, X + delta , X_denoise + delta)
        else:
            try:
                criterion_name = criterion.func.__name__
                if criterion_name == 'second_order':
                    loss = criterion(model, X + delta, y)
                else:
                    loss = criterion(model(X + delta), y)
            except:
                loss = criterion(model(X + delta), y)

        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X_denoise + delta).clamp(*bound) - X_denoise
        delta.grad.zero_()

    delta.data = (X_denoise + delta).clamp(*bound) - X_denoise
    return delta.detach()

# def pgd_linf_1_a(model, criterion, X, X_denoise,a,y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, randomize=False):
#     """ Construct PGD adversarial examples on the examples X"""
#     if randomize:
#         delta = torch.rand_like(X_denoise, requires_grad=True)
#         delta.data = delta.data * 2 * epsilon - epsilon
#     else:
#         delta = torch.zeros_like(X_denoise, requires_grad=True)
#
#     for t in range(num_iter):
#         if y is None:
#             # print("t:{}".format(t))
#             # print("X+delta.shape:{}".format((X+delta).shape))
#             # loss = criterion(model,X + delta,X_denoise + delta)
#             loss = criterion(model, X + delta , X_denoise + delta,a=a)
#         else:
#             try:
#                 criterion_name = criterion.func.__name__
#                 if criterion_name == 'second_order':
#                     loss = criterion(model, X + delta, y)
#                 else:
#                     loss = criterion(model(X + delta), y)
#             except:
#                 loss = criterion(model(X + delta), y)
#
#         loss.backward()
#         delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.data = (X_denoise + delta).clamp(*bound) - X_denoise
#         delta.grad.zero_()
#
#     delta.data = (X_denoise + delta).clamp(*bound) - X_denoise
#     return delta.detach()
#
#
# def pgd_linf_2_a(model, criterion, X,a, y=None,epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, randomize=False):
#     """ Construct PGD adversarial examples on the examples X"""
#     if randomize:
#         delta = torch.rand_like(X, requires_grad=True)
#         delta.data = delta.data * 2 * epsilon - epsilon
#     else:
#         delta = torch.zeros_like(X, requires_grad=True)
#
#     for t in range(num_iter):
#         if y is None:
#             # print("t:{}".format(t))
#             # print("X+delta.shape:{}".format((X+delta).shape))
#             loss = criterion(model, X + delta,a)
#         else:
#             try:
#                 criterion_name = criterion.func.__name__
#                 if criterion_name == 'second_order':
#                     loss = criterion(model, X + delta, y)
#                 else:
#                     loss = criterion(model(X + delta), y)
#             except:
#                 loss = criterion(model(X + delta), y)
#
#         if loss==0:
#             delta.data = (X + delta).clamp(*bound) - X
#             return delta.detach()
#             break
#         else:
#             loss.backward()
#             delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#             delta.data = (X + delta).clamp(*bound) - X
#             delta.grad.zero_()
#
#     delta.data = (X + delta).clamp(*bound) - X
#     return delta.detach()

def bpda(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, purify=None):

    delta = torch.zeros_like(X)
    for t in range(num_iter):

        X_pfy = purify(model, X=X + delta).detach()
        X_pfy.requires_grad_()

        loss = criterion(model(X_pfy), y)
        loss.backward()

        delta.data = (delta + step_size*X_pfy.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        X_pfy.grad.zero_()

    return delta.detach()

def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()

def apgd_tta(model, x, y, norm, eps, n_iter=10, use_rs=False, loss='ce',
               verbose=False, is_train=True, use_interm=False):
    assert not model.training
    device = x.device
    ndims = len(x.shape) - 1


    x_adv = x.clone()
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)

    # set loss
    criterion_indiv = nn.CrossEntropyLoss(reduction='none')

    # set pa rams
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.

    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims],
                                         device=device)
    counter3 = 0

    # x_adv.requires_grad_()
    grad = torch.zeros_like(x)
    # for _ in range(self.eot_iter)
    # with torch.enable_grad()

    logits, interm_x = model(x_adv)
    for x_step in interm_x:
        x_step.requires_grad_(True)
        logits = model.model(x_step)
        loss_indiv = criterion_indiv(logits, y)
        loss = loss_indiv.sum()
        grad += torch.autograd.grad(loss, [x_step])[0].detach()
        # x_step.detach_()
        loss_indiv.detach_()
        loss.detach_()
    assert not x_adv.requires_grad
    grad /= interm_x.shape[0]

    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()

    for i in range(n_iter):
        ### gradient step
        if True:  # with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach().mean()

            a = 0.75 if i > 0 else 1.0

            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),x - eps), x + eps), 0.0, 1.0)
            x_adv = x_adv_1 + 0.



        logits, interm_x = model(x_adv)
        grad = torch.zeros_like(x)
        for x_step in interm_x:
            x_step.requires_grad_(True)
            logits = model.model(x_step)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_step])[0].detach()
            # x_step.detach_()
            loss_indiv.detach_()
            loss.detach_()
        assert not x_adv.requires_grad
        grad /= interm_x.shape[0]

        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        if verbose:
            str_stats = ' - step size: {:.5f}'.format(step_size.mean())
            print('iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            # print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))

        ### check step size
        if True:  # with torch.no_grad()
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1
            if counter3 == k:
                if norm in ['Linf', 'L2']:
                    fl_oscillation = check_oscillation(loss_steps, i, k,
                                                       loss_best, k3=thr_decr)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                                               fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    counter3 = 0
                    k = max(k - size_decr, n_iter_min)

    return x_best, acc, loss_best, x_best_adv


def cw(model, criterion, X, y=None, epsilon=0.1, num_classes=10):
    delta = L2Adversary()(model, X.clone().detach(), y, num_classes=num_classes).to(X.device) - X
    delta_norm_1 = torch.norm(delta, p=2, dim=1, keepdim=True)
    delta_norm_2 = torch.norm(delta_norm_1, p=2, dim=2, keepdim=True)
    delta_norm = torch.norm(delta_norm_2, p=2, dim=3, keepdim=True)
    # delta_norm = torch.norm(delta, p=2, dim=(1,2,3),keepdim=True) + 1e-4
    delta_proj = (delta_norm > epsilon).float() * delta / delta_norm * epsilon + (delta_norm < epsilon).float() * delta
    return delta_proj

def df(model, criterion, X, y=None, epsilon=0.1):
    delta = DeepFool()(model, X.clone().detach()).clamp(0,1) - X
    # print(delta.shape)
    delta_norm_1 = torch.norm(delta, p=2, dim=1,keepdim=True)
    # print(delta_norm_1.shape)
    delta_norm_2=torch.norm(delta_norm_1, p=2, dim=2,keepdim=True)
    # print(delta_norm_2.shape)
    delta_norm=torch.norm(delta_norm_2, p=2, dim=3,keepdim=True)
    # print(delta_norm.shape)
    delta_proj = (delta_norm > epsilon).float() * delta / delta_norm * epsilon + (delta_norm < epsilon).float() * delta
    return delta_proj
