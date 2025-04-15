import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad, Variable

from attacks import *
# from defenses import *
from criterions import *
from entropy_shot import *

import os
import copy
import pickle
import numpy as np
from collections import deque
# import advertorch
from autoattack import AutoAttack
from torchvision import transforms
from datetime import datetime



toImage = transforms.ToPILImage()


def train(model, train_loader, criterion, optimizer, scheduler, device):
    '''
    scheduler not used
    '''
    model.train()
    error, acc = 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        error += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(train_loader)
    acc = acc / len(train_loader.dataset)
    print('train loss: {} / acc: {}'.format(error, acc))


def train_with_auxiliary(model, train_loader, joint_criterion, optimizer, scheduler, device):
    model.train()
    error, acc, error_aux, acc_aux = 0., 0., 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        joint_loss, (loss, aux_loss) = joint_criterion(model, X=X, y=y)
        error += loss.item()
        error_aux += aux_loss.item()

        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()

        acc += (model.pred[:y.shape[0]].max(dim=1)[1] == y).sum().item()
        if joint_criterion.keywords['aux_criterion'].__name__ == 'rotate_criterion':
            acc_aux += (model.pred_deg.max(dim=1)[1].cpu() == torch.arange(4)[:, None].repeat(1, X.shape[
                0]).flatten()).sum().item()

    error = error / len(train_loader)
    error_aux = error_aux / len(train_loader)
    acc = acc / len(train_loader.dataset)
    if joint_criterion.keywords['aux_criterion'].__name__ == 'rotate_criterion':
        acc_aux = acc_aux / len(train_loader.dataset) / 4
        print('train loss: {} / acc: {} / err-aux: {} / acc-aux: {}'.format(error, acc, error_aux, acc_aux))
    else:
        print('train loss: {} / acc: {} / err-aux: {}'.format(error, acc, error_aux))


def train_adversarial(model, train_loader, criterion, attack, optimizer, device):
    model.train()

    error, acc = 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        if attack is not None:
            model.eval()
            delta = attack(model, criterion, X, y)
            model.train()
            pred = model(X + delta)
        else:
            pred = model(X)

        loss = criterion(pred, y)
        error += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc += (pred.max(dim=1)[1] == y).sum().item()
    error = error / len(train_loader)
    acc = acc / len(train_loader.dataset)
    print('train loss: {} / acc: {}'.format(error, acc))


def evaluate(model, eval_loader, criterion, device):
    model.eval()

    error, acc = 0., 0.
    with torch.no_grad():
        for X, y in eval_loader:
            X, y = X.to(device), y.to(device)
            # print(X.shape)
            # print(y.shape)
            pred = model(X)
            # print(pred.shape)

            loss = criterion(pred, y)
            # print(loss)
            error += loss.item()
            # print(error)

            acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(eval_loader)
    acc = acc / len(eval_loader.dataset)
    print('val loss: {} / acc: {}'.format(error, acc))

    return acc


def evaluate_auxiliary(model, eval_loader, aux_criterion, device):
    model.eval()
    error, acc = 0., 0.
    with torch.no_grad():
        for X, y in eval_loader:
            X, y = X.to(device), y.to(device)
            loss = aux_criterion(model, X)
            error += loss.item()
    error = error / len(eval_loader)
    # print('val loss: {}'.format(error))


# 将 N x C x H X W 的tensor格式图片转化为相应的numpy格式
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((0, 2, 3, 1))
    return img


# 将 N x H x W X C 的numpy格式图片转化为相应的tensor格式
def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img.float().div(255)


def elapsed_seconds(start_time, end_time):
    dt = end_time - start_time

    return (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1000000.0


def process_denoise_and_adv(  ml1, ml2, ml4,X_pfy_denoise, X_pfy_denoise_before,X_pfy_denoise_last, X, delta, device):
    """
    处理去噪张量并生成对抗样本。

    参数:
        ml1 (Tensor): 标签反转布尔张量。
        ml2 (Tensor): 熵阈值布尔张量。
        ml4 (Tensor): 其他布尔条件张量。
        X_pfy_denoise (Tensor): 当前的去噪样本张量。
        X_pfy_denoise_before (Tensor): 用于存储上一步去噪样本的张量。
        X_pfy_denoise_last (Tensor): 用于存储当前符合条件的最终样本张量。
        X (Tensor): 原始输入样本张量。
        delta (Tensor): 对抗扰动张量。
        device (torch.device): 计算设备。

    返回:
        Tensor: 更新后的 `X_pfy_denoise_last`。
        Tensor: 更新后的 `X_pfy_denoise_next`。
        Tensor: 更新后的对抗样本张量 `Xadv`。
    """
    # 生成布尔张量 ml3
    ml3 = torch.tensor([(i or j) and h for i, j, h in zip(ml2, ml1, ml4)], dtype=torch.bool,device=device)

    # 初始化结果张量
    X_pfy_denoise_next = torch.zeros_like(X_pfy_denoise).cuda(device)
    Xadv = torch.zeros_like(X + delta).cuda(device)

    is_zero = (X_pfy_denoise_before == 0).to(device)

    # 判断哪些样本是全零
    # 展平除第一个维度之外的所有维度
    is_zero = is_zero.view(is_zero.size(0), -1).all(dim=1)

    not_zero = ~is_zero

    # 根据条件更新张量
    condition_last = not_zero & ml3
    condition_next = not_zero & ~ml3

    X_pfy_denoise_last[condition_last] = X_pfy_denoise[condition_last]
    X_pfy_denoise_next[condition_next] = X_pfy_denoise[condition_next]
    Xadv[condition_next] = (X + delta)[condition_next]

    return X_pfy_denoise_last, X_pfy_denoise_next, Xadv

def evaluate_adversarial(model, loader, criterion, aux_criterion_1, aux_criterion_2, attack, purify, device, args,sub_model=None):
    model.eval()
    acc=0.

    error_cln, acc_cln = 0., 0.
    error_adv, acc_adv = 0., 0.
    error_pfy, acc_pfy = 0., 0.


    if sub_model != None:
        # FGSM_T = advertorch.attacks.GradientSignAttack(predict=sub_model.to(device),
        #                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        #                                                targeted=True)
        # PGD_T = LinfPGDAttack(
        #     sub_model.to(device), loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        #     nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
        # AA_N = AutoAttack(sub_model.to(device), norm='Linf', eps=0.3, version='standard', device=device)

        # FGSM_T = advertorch.attacks.GradientSignAttack(predict=sub_model.to(device),
        #                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8 / 255,
        #                                                targeted=True)
        # PGD_T = LinfPGDAttack(
        #     sub_model.to(device), loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8 / 255,
        #     nb_iter=20, eps_iter=2 / 255, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

        AA_N = AutoAttack(sub_model.to(device), norm='Linf', eps=8 / 255, version='standard', device=device)

    else:
        # FGSM_T = advertorch.attacks.GradientSignAttack(predict=model.to(device),
        #                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        #                                                targeted=True)

        # FGSM_T = advertorch.attacks.GradientSignAttack(predict=model.to(device),
        #                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
        #                                                targeted=True)
        #
        # PGD_T = LinfPGDAttack(
        # model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
        # nb_iter=20, eps_iter=2/255, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

        # PGD_T = LinfPGDAttack(
        #     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        #     nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

        AA_N = AutoAttack(model, norm='Linf', eps=8 / 255, version='standard', device=device)

        # AA_N = AutoAttack(model, norm='Linf', eps=0.3, version='standard',device=device)

    #检测cln/adv样本
#     resnet-pi cifar100
#     laux_1=0.2
#     ent_1=0.2
#     laux_2=0.2
#     ent_2=0.2

    # widresnet-pi cifar100
    # laux_1=0.02
    # ent_1=0.1
    # laux_2=0.04
    # ent_2=0.1
    # widresnet-pi cifar100 test
    # laux_1=0.02
    # ent_1=0.1
    # laux_2=0.04
    # ent_2=0.1

#   resnet rot  cifar10
    # laux_1=0.
    # ent_1=0.1
    # laux_2=0.03
    # ent_2=0.01
    
#  ae mnist rec  
#     laux_1=0.007
#     ent_1=0.1
#     laux_2=0.007
#     ent_2=0.01

    #  cae mnist rec
    # laux_1=0.02
    # ent_1=0.1
    # laux_2=0.02
    # ent_2=0.01

    #  resnet cifar10 rec USE LAST
    # laux_1 = 0
    # ent_1 = 0.1
    # laux_2 = 0.015
    # ent_2 = 0.1

    # resnet cifar10 rec nonoise
    # laux_1 = 0.02
    # ent_1 = 0.05
    # laux_2 = 0.02
    # ent_2 = 0.05

    #  resnet cifar100 rec
    # laux_1 = 0.
    # ent_1 = 0.1
    # laux_2 = 0.03
    # ent_2 = 0.1



#     resnet-pi cifar10
    laux_1=0.157
    ent_1=0.23
    laux_2=0.157
    ent_2=0.2

    # laux_1 = 0
    # ent_1 = 0.23
    # laux_2 = 0
    # ent_2 = 0.2

    #     WIDresnet-pi cifar10
    # laux_1 = 0.01
    # ent_1 = 0.03
    # laux_2 = 0.03
    # ent_2 = 0.03

    #  widresnet cifar10 rec
    # laux_1 = 0
    # ent_1 = 0.1
    # laux_2 = 0.016
    # ent_2 = 0.05

    #  widresnet cifar100 rec
    # laux_1 = 0
    # ent_1 = 0.1
    # laux_2 = 0.018
    # ent_2 = 0.08

    # resnet imagenet lc
    # laux_1 = 0
    # ent_1 = 0.1
    # laux_2 = 0.09
    # ent_2 = 0.15
    # resnet imagenet lc test
    # laux_1 = 0.2
    # ent_1 = 0.2
    # laux_2 = 0.2
    # ent_2 = 0.2


    # widresnet imagenet lc
    # laux_1 = 0.09
    # ent_1 = 0.08
    # laux_2 = 0.05
    # ent_2 = 0.1

    attack_time_all=0
    defense_time_all=0
    for num_attack in range(4):
        if num_attack==0:
            continue
            # print('AA attack blak box')

        elif num_attack == 1:
            continue
            # print('FGSM target attack blak box')

        elif num_attack == 2:
            continue
            # print('PGD target attack blak box')

        for i, (X, y) in enumerate(loader):

            X, y = X.to(device), y.to(device)

            # if 'model' in attack.keywords.keys():  # if substitute model is specified
            #     delta = attack(criterion=criterion, X=X, y=y)
            start_time = datetime.now()
            if sub_model != None:  # if substitute model is specified
                delta = attack(criterion=criterion, X=X, y=y)
            else:
                delta = attack(model, criterion, X, y)
            X_adv = delta + X
            attack_time = elapsed_seconds(start_time, datetime.now())
            # if i%100==0:
            #     print("[{}] Epoch {}:\t{:.2f} seconds to attack {} data".format(str(datetime.now()), i, attack_time, X.shape[0]))
            attack_time_all+=attack_time

            delta = X_adv - X



            X_pfy_denoise_last_0 = torch.zeros_like(X + delta).cuda(device)
            X_pfy_denoise_last_1 = torch.zeros_like(X + delta).cuda(device)
            X_pfy_denoise_last_2 = torch.zeros_like(X + delta).cuda(device)
            X_pfy_denoise_last_3 = torch.zeros_like(X + delta).cuda(device)
            X_pfy_denoise_last_4 = torch.zeros_like(X + delta).cuda(device)
            X_pfy_denoise_next_0 = torch.zeros_like(X + delta).cuda(device)



            with torch.no_grad():
                if args.auxiliary == 'rec':
                    ml1, pred_x = laux_num_rec(model,X + delta, X + delta, train=False,laux=laux_1)  # 判断laux阈值
                    ml2 = ent_num(pred_x,ent=ent_1)  # 预测熵阈值

                elif args.auxiliary == 'pi':
                    ml1, pred_x = laux_num_lc(model, X + delta, train=False,laux=laux_1)  # 判断laux阈值
                    ml2 = ent_num(pred_x,ent=ent_1)  # 预测熵阈值
            # detect_time = elapsed_seconds(start_time, datetime.now())
            # print("[{}] Epoch {}:\t{:.2f} seconds to detect {} data".format(str(datetime.now()), i, detect_time, X.shape[0]))


            # 将 ml3 转换为布尔张量
            ml3 = torch.tensor([i and j for i, j in zip(ml2, ml1)], dtype=torch.bool)
            # 根据 ml3 进行布尔索引操作
            X_pfy_denoise_last_0[ml3] = (X+delta)[ml3]
            X_pfy_denoise_next_0[~ml3] = (X+delta)[~ml3]


            #纯化
            start_time = datetime.now()
            if args.auxiliary == 'rec':
                X_pfy = purify(model, aux_criterion_2, X=X_pfy_denoise_next_0, X_denoise=X_pfy_denoise_next_0)
            else :
                X_pfy = purify(model, aux_criterion_2, X=X_pfy_denoise_next_0)

            zero_mask = (X_pfy_denoise_next_0 == 0).all(dim=1).all(dim=1).all(dim=1)
            X_pfy[zero_mask] = torch.zeros_like(X_pfy[0]).to(X_pfy.device)


            # print("start x_pfy_2")
            if args.auxiliary == 'rec':
                X_pfy_denoise = purify(model, aux_criterion_1, X=X_pfy_denoise_next_0, X_denoise=X_pfy)
            else :
                X_pfy_denoise = purify(model, aux_criterion_1, X=X_pfy)

            X_pfy_denoise[zero_mask] = torch.zeros_like(X_pfy_denoise[0]).to(X_pfy_denoise.device)

            with torch.no_grad():
                if args.auxiliary == 'rec':
                    ml4, pred_pfy_denoise = laux_num_rec(model,X_pfy_denoise_next_0, X_pfy_denoise, train=False,laux=laux_2)
                elif args.auxiliary == 'rot':
                    ml4, pred_pfy_denoise = laux_num_rot(model, X_pfy_denoise, train=False,laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'pi':
                    ml4, pred_pfy_denoise = laux_num_lc(model, X_pfy_denoise, train=False,laux=laux_2)


            ml1 = label_reverse(pred_x, pred_pfy_denoise)#判断标签反转
            ml2 = ent_num(pred_pfy_denoise,ent=ent_2)  # 预测熵阈值

            X_pfy_denoise_last_1,X_pfy_denoise_next_1,Xadv=process_denoise_and_adv(ml1,ml2,ml4,X_pfy_denoise,X_pfy_denoise_next_0,X_pfy_denoise_last_1,X,delta,X_pfy_denoise.device)

            if args.auxiliary == 'rec':
                X_pfy_denoise_3 = purify(model, aux_criterion_2, X=Xadv, X_denoise=X_pfy_denoise_next_1)
            else :
                X_pfy_denoise_3 = purify(model, aux_criterion_2, X=X_pfy_denoise_next_1)

            zero_mask = (X_pfy_denoise_next_1 == 0).all(dim=1).all(dim=1).all(dim=1)
            X_pfy_denoise_3[zero_mask] = torch.zeros_like(X_pfy_denoise_3[0]).to(X_pfy.device)

            if args.auxiliary == 'rec':
                X_pfy_denoise_4 = purify(model, aux_criterion_1, X=Xadv, X_denoise=X_pfy_denoise_3)
            else :
                X_pfy_denoise_4 = purify(model, aux_criterion_1, X=X_pfy_denoise_3)

            X_pfy_denoise_4[zero_mask] = torch.zeros_like(X_pfy_denoise_4[0]).to(X_pfy.device)


            with torch.no_grad():
                if args.auxiliary == 'rec':
                    ml4, pred_pfy_denoise_4 = laux_num_rec(model,Xadv, X_pfy_denoise_4, train=False,laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'rot':
                    ml4, pred_pfy_denoise_4 = laux_num_rot(model, X_pfy_denoise_4, train=False,laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'pi':
                    ml4, pred_pfy_denoise_4 = laux_num_lc(model, X_pfy_denoise_4, train=False,laux=laux_2)


            ml1 = label_reverse(pred_x, pred_pfy_denoise_4)  # 判断标签反转
            ml2 = ent_num(pred_pfy_denoise_4, ent=ent_2)  # 预测熵阈值

            X_pfy_denoise_last_2,X_pfy_denoise_next_2,Xadv=process_denoise_and_adv(ml1,ml2,ml4,X_pfy_denoise_4,X_pfy_denoise_next_1,X_pfy_denoise_last_2,X,delta,device)


            if args.auxiliary == 'rec':
                X_pfy_denoise_5 = purify(model, aux_criterion_2, X=Xadv, X_denoise=X_pfy_denoise_next_2)
            else :
                X_pfy_denoise_5 = purify(model, aux_criterion_2, X=X_pfy_denoise_next_2)
            zero_mask = (X_pfy_denoise_next_2 == 0).all(dim=1).all(dim=1).all(dim=1)

            X_pfy_denoise_5[zero_mask] = torch.zeros_like(X_pfy_denoise_5[0]).to(X_pfy.device)


            # print('start x_pfy_6')
            if args.auxiliary == 'rec':
                X_pfy_denoise_6 = purify(model, aux_criterion_1, X=Xadv, X_denoise=X_pfy_denoise_5)
            else :
                X_pfy_denoise_6 = purify(model, aux_criterion_1, X=X_pfy_denoise_5)

            X_pfy_denoise_6[zero_mask] = torch.zeros_like(X_pfy_denoise_6[0]).to(X_pfy.device)

            with torch.no_grad():
                if args.auxiliary == 'rec':
                    ml4, pred_pfy_denoise_6 = laux_num_rec(model, Xadv,X_pfy_denoise_6, train=False,laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'rot':
                    ml4, pred_pfy_denoise_6 = laux_num_rot(model, X_pfy_denoise_6, train=False,laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'pi':
                    ml4, pred_pfy_denoise_6 = laux_num_lc(model, X_pfy_denoise_6, train=False,laux=laux_2)

            ml1 = label_reverse(pred_x, pred_pfy_denoise_6)  # 判断标签反转
            ml2 = ent_num(pred_pfy_denoise_6, ent=ent_2)  # 预测熵阈值

            X_pfy_denoise_last_3,X_pfy_denoise_next_3,Xadv=process_denoise_and_adv(ml1,ml2,ml4,X_pfy_denoise_6,X_pfy_denoise_next_2,X_pfy_denoise_last_3,X,delta,device)

            if args.auxiliary == 'rec':
                X_pfy_denoise_7 = purify(model, aux_criterion_2, X=Xadv, X_denoise=X_pfy_denoise_next_3)
            else :
                X_pfy_denoise_7 = purify(model, aux_criterion_2, X=X_pfy_denoise_next_3)
            zero_mask = (X_pfy_denoise_next_3 == 0).all(dim=1).all(dim=1).all(dim=1)
            X_pfy_denoise_7[zero_mask] = torch.zeros_like(X_pfy_denoise_7[0]).to(X_pfy.device)


            # print('start x_pfy_8')
            if args.auxiliary == 'rec':
                X_pfy_denoise_8 = purify(model, aux_criterion_1, X=Xadv, X_denoise=X_pfy_denoise_7)
            else :
                X_pfy_denoise_8 = purify(model, aux_criterion_1, X=X_pfy_denoise_7)
            X_pfy_denoise_8[zero_mask] = torch.zeros_like(X_pfy_denoise_8[0]).to(X_pfy.device)

            with torch.no_grad():
                if args.auxiliary == 'rec':
                    ml4, pred_pfy_denoise_8 = laux_num_rec(model,Xadv, X_pfy_denoise_8, train=False, laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'rot':
                    ml4, pred_pfy_denoise_8 = laux_num_rot(model, X_pfy_denoise_8, train=False,laux=laux_2)  # 判断laux阈值
                elif args.auxiliary == 'pi':
                    ml4, pred_pfy_denoise_8 = laux_num_lc(model, X_pfy_denoise_8, train=False, laux=laux_2)  # 判断laux阈值

            ml1 = label_reverse(pred_x, pred_pfy_denoise_8)  # 判断标签反转
            ml2 = ent_num(pred_pfy_denoise_8, ent=ent_2)  # 预测熵阈值

            X_pfy_denoise_last_4,X_pfy_denoise_next_4,Xadv=process_denoise_and_adv(ml1,ml2,ml4,X_pfy_denoise_8,X_pfy_denoise_next_3,X_pfy_denoise_last_4,X,delta,device)


            # print('start x_pfy_9')
            if args.auxiliary == 'rec':
                X_pfy_denoise_9 = purify(model, aux_criterion_2, X=Xadv, X_denoise=X_pfy_denoise_next_4)
            else :
                X_pfy_denoise_9 = purify(model, aux_criterion_2, X=X_pfy_denoise_next_4)

            zero_mask = (X_pfy_denoise_next_4 == 0).all(dim=1).all(dim=1).all(dim=1)
            X_pfy_denoise_9[zero_mask] = torch.zeros_like(X_pfy_denoise_8[0]).to(X_pfy.device)

            # print('start x_pfy_10')
            if args.auxiliary == 'rec':
                X_pfy_denoise_10 = purify(model, aux_criterion_1, X=Xadv, X_denoise=X_pfy_denoise_9)
            else :
                X_pfy_denoise_10 = purify(model, aux_criterion_1, X=X_pfy_denoise_9)

            X_pfy_denoise_10[zero_mask] = torch.zeros_like(X_pfy_denoise_10[0]).to(X_pfy.device)
            pred_pfy_denoise_10=model(X_pfy_denoise_10)
            ml2 = ent_num(pred_pfy_denoise_10,ent=ent_2)

            X_pfy_denoise_last_5=X_pfy_denoise_10

            X_pfy_denoise_last=X_pfy_denoise_last_0+X_pfy_denoise_last_1+X_pfy_denoise_last_2+X_pfy_denoise_last_3+X_pfy_denoise_last_4+X_pfy_denoise_last_5
            defense_time = elapsed_seconds(start_time, datetime.now())
            # if i%100==0:
            #     print("[{}] Epoch {}:\t{:.2f} seconds to defense {} data".format(str(datetime.now()), i, defense_time, X.shape[0]))
            defense_time_all+=defense_time

            with torch.no_grad():

                pred_cln = model(X)
                pred_adv = model(X + delta)
                pred = model(X_pfy_denoise_last)

                loss_cln = nn.functional.cross_entropy(pred_cln, y)
                error_cln += loss_cln.item()
                acc_cln += (pred_cln.max(dim=1)[1] == y).sum().item()

                loss_adv = nn.functional.cross_entropy(pred_adv, y)
                error_adv += loss_adv.item()
                acc_adv += (pred_adv.max(dim=1)[1] == y).sum().item()

                loss = nn.functional.cross_entropy(pred, y)
                error_pfy += loss.item()
                acc_pfy += (pred.max(dim=1)[1] == y).sum().item()


        error_cln= error_cln / len(loader)
        acc_cln = acc_cln / len(loader.dataset)

        error_adv = error_adv / len(loader)
        acc_adv = acc_adv / len(loader.dataset)

        error_pfy=error_pfy / len(loader)
        acc_pfy = acc_pfy / len(loader.dataset)


        # print('原防御成功率:{}% '.format(acc_adv*100))
        # print('REAL防御成功率:{}% '.format(acc_pfy*100))
        # print('提升：{}%'.format((acc_pfy_10-acc_adv)*100))acc_pfy

        # print('原baseline防御成功率:{}% '.format(acc_adv * 100))
        print('在FGSM攻击下的鲁棒性防御成功率为：{:.2f}%'.format((acc_pfy)*100))

    return acc


def evaluate_at_adversarial(model, loader, criterion, attack, device):
    model.eval()
    error, acc = 0., 0.
    clean, adv, df = 0., 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if 'model' in attack.keywords.keys():  # if substitute model is specified
            delta = attack(criterion=criterion, X=X, y=y)
        else:
            delta = attack(model, criterion, X, y)

        pred = model(X + delta)
        loss = nn.functional.cross_entropy(pred, y)
        error += loss.item()
        acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(loader)
    acc = acc / len(loader.dataset)
    print('adv loss: {} / acc: {}'.format(error, acc))

    return acc


def evaluate_bpda_adversarial_soap(model, x_adv,y_test, criterion, aux_criterion,  purify, device):
    model.eval()
    error, acc = 0., 0.
    error_pfy_1=0.
    error_aux_adv=0.
    clean, adv, df = 0., 0., 0.
    batch_size = 100
    n_batches = math.ceil(x_adv.shape[0] / batch_size)
    lent_adv = 0
    lent_pfy_1 = 0
    # laux_1 = 0.157
    # ent_1 = 0.23
    # laux_2 = 0.157
    # ent_2 = 0.2
    allnum_adv = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    allnum_pred = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    accnum_adv = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    accnum_pred = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    lent_adv = 0
    lent_pfy_1 = 0
    error_adv, acc_adv = 0., 0.
    error_pfy, acc_pfy = 0., 0.

    for counter in range(n_batches):
        X = x_adv[counter * batch_size:(counter + 1) *
                                       batch_size].to(device)
        y = y_test[counter * batch_size:(counter + 1) * batch_size].to(device)

        X, y = X.to(device), y.to(device)

        X_pfy = purify(model, aux_criterion, X)

        with torch.no_grad():
            # aux_loss_adv, l = pi_criterion(model, X, joint=True, train=False)
            # aux_loss_1, l = pi_criterion(model, X_pfy, joint=True, train=False)
            pred_adv=model(X)
            label_pred_adv = predict_from_logits(pred_adv)
            pred_pfy = model(X_pfy)
            label_pred_pfy = predict_from_logits(pred_pfy)


            loss = nn.functional.cross_entropy(pred_pfy, y)
            error += loss.item()
            acc += (pred_pfy.max(dim=1)[1] == y).sum().item()
            acc_adv += (pred_adv.max(dim=1)[1] == y).sum().item()


    error = error / n_batches
    acc = acc / 1000
    acc_adv = acc_adv / 1000
    error_pfy_1 = error_pfy_1 / n_batches
    error_aux_adv = error_aux_adv / n_batches
    print('adv loss: {} / acc: {}'.format(error, acc))
    print('acc_adv: {}'.format(acc_adv))



    return acc


def save_reps(model, train_loader, criterion, attack, defense, save_dir, device):
    if os.path.exists(os.path.join(save_dir, 'reps.pkl')):
        print('representation already exists!')
        return
    model.eval()
    reps, labels = [], []

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        if attack is None:
            reps.append(model(X, return_reps=True))
        else:
            delta = attack(model, criterion, X, y)
            if defense is not None:
                inv_delta = defense(model, X=X + delta)
                reps.append(model((X + delta + inv_delta).clamp(0, 1), return_reps=True))
            else:
                reps.append(model(X + delta, return_reps=True))
        labels.append(y)
    reps = torch.cat(reps, dim=0).detach()
    labels = torch.cat(labels, dim=0)
    if save_dir is None:
        return reps, labels
    with open(os.path.join(save_dir, 'reps_pfy.pkl'), 'wb') as f:
        pickle.dump(reps.cpu().numpy(), f)


def save_logits(model, train_loader, save_dir, device):
    if os.path.exists(os.path.join(save_dir, 'logits.pkl')):
        return
    model.eval()
    logits = []
    with torch.no_grad():
        for X, _ in train_loader:
            X = X.to(device).flatten(start_dim=1)
            logits.append(model(X))
    logits = torch.cat(logits, dim=0).cpu().numpy()
    with open(os.path.join(save_dir, 'logits.pkl'), 'w') as f:
        pickle.dump(logits, f)


def save_file(buffer, file_dir):
    buffer = torch.cat(buffer, dim=0).cpu().numpy()
    with open(file_dir, 'wb') as f:
        pickle.dump(buffer, f)


def jacobian_augment(model, train_loader, lmbda, device):
    model.eval()
    new_data = []
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X.requires_grad_()
        l = model(X)
        loss = l[torch.arange(y.shape[0]), y].sum()
        loss.backward()
        new_data.append(X + lmbda * X.grad.sign())
    return torch.cat(new_data, dim=0).detach().cpu()
