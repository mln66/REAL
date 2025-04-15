import os
import torch
import torch.nn as nn
import numpy as np
import math


def num_shot(x):

    loss_ent = x.softmax(1)
    # print(loss_ent)
    num_class=x.shape[1]
    # print('num_class{}'.format(num_class))
    m = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1)/math.log(num_class)

    loss_ent = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1).mean(0)/math.log(num_class)

    # print(loss_ent)
    ls=[]
    for iterm in m:
        if iterm<=0.1:
            ls.append(0)
        elif iterm<=0.2:
            ls.append(1)
        elif iterm<=0.3:
            ls.append(2)
        elif iterm<=0.4:
            ls.append(3)
        elif iterm<=0.5:
            ls.append(4)
        elif iterm<=0.6:
            ls.append(5)
        elif iterm<=0.7:
            ls.append(6)
        elif iterm<=0.8:
            ls.append(7)
        elif iterm<=0.9:
            ls.append(8)
        else:
            ls.append(9)
    # print(ls)
    # loss_div = torch.sum(loss_div * torch.log(loss_div + 1e-5))
    # return loss_ent + loss_div
    return loss_ent,ls


def allnum(x, ls):
    for i, item in enumerate(ls):
        x[str(item)] += 1
    return x

def accnum(true, pred_adv, dic, ls):
    pred_adv_1=pred_adv
    pred_adv_1=pred_adv_1.cpu().detach()
    true_1=true
    true_1=true_1.cpu().detach()
    # print(pred_adv_1)
    ml = np.array(true_1) == np.array(pred_adv_1)
    # print(ml)
    for i, item in enumerate(ml):
        if item:
            dic[str(ls[i])] += 1
    return dic

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def label_reverse(pred_adv,pred_pfy):
    pred_adv_1 = pred_adv
    pred_adv_1 = pred_adv_1.cpu().detach()
    pred_pfy_1 = pred_pfy
    pred_pfy_1 = pred_pfy_1.cpu().detach()
    ml1=np.array(pred_pfy_1.max(dim=1)[1]) != np.array(pred_adv_1.max(dim=1)[1])
    return ml1


def laux_num_rec_batch(model,X,X_denoise,laux,train=False):
    l = model(X_denoise, add_noise=train)
    loss_2 = nn.functional.mse_loss(model.r, X)
    print(loss_2)
    # print('loss2.shape:{}'.format(loss_2.shape))
    # loss_1 = torch.mean(loss_2, dim=1)
    # loss_1 = torch.mean(loss_1, dim=1)
    # loss_1 = torch.mean(loss_1, dim=1)
    ml1=loss_2<=laux
    return ml1, l

def laux_num_rec(model,X,X_denoise,laux,train=False):
    l = model(X_denoise, add_noise=train)
    loss_2 = nn.functional.mse_loss(model.r, X, reduction='none')
    # print('loss2.shape:{}'.format(loss_2.shape))
    loss_1 = torch.mean(loss_2, dim=1)
    loss_1 = torch.mean(loss_1, dim=1)
    loss_1 = torch.mean(loss_1, dim=1)

    ml1=loss_1<=laux
    return ml1, l

def laux_num_lc(model,X,laux,train=False):
    X1 = X
    if X.shape[-1]==64:
        X2 = random_trans_combo_tiny(X,df=~train)
    else:
        X2 = random_trans_combo(X,df=~train)
    l1, l2 = model(X1), model(X2)
    # print(l1.shape)
    l = torch.cat((l1, l2), dim=0)
    loss_2 = nn.functional.mse_loss(l1, l2,reduction='none')
    # print(loss_2.shape)
    loss_1 = torch.mean(loss_2, dim=1)
    # print(loss_1.shape)
    ml1=loss_1<=laux
    return ml1, l1

def laux_num_rot(model,X,laux,train=False):
    l=model(X, add_noise=train)
    batch_size=l.shape[0]
    a = torch.empty(batch_size, 1)
    loss_total = torch.zeros_like(a)
    X_rotated = []
    for deg in [0, 90, 180, 270]:
        X_rotated.append(rotate_images(X, deg))
    X_append = torch.cat(X_rotated, dim=0)
    l_deg = model(X_append, add_noise=train)
    # l = model(X)
    y_deg = torch.arange(4)[:, None].repeat(1, X_append.shape[0]//4).flatten().to(X_append.device)
    onehot = torch.zeros(X_append.shape[0], 4).to(X_append.device)
    onehot[torch.arange(X_append.shape[0]), y_deg] = 1
    # print('onehot:{}'.format(onehot))
    # print('onehot.shape:{}'.format(onehot.shape))
    loss_2 = nn.functional.mse_loss(torch.softmax(model.pred_deg, dim=1), onehot,reduction='none')
    # print('pred_deg_loss.shape:{}'.format(loss_2.shape))
    
    loss_1 = torch.mean(loss_2, dim=1)
    list1=loss_1[0:len(loss_1):4]
    list2=loss_1[1:len(loss_1):4]
    list3=loss_1[2:len(loss_1):4]
    list4=loss_1[3:len(loss_1):4]
    loss_1=(list1+list2+list3+list4)/4
  
    for i ,item in enumerate(X):
        if not torch.equal(item.cpu(), torch.zeros(item.shape)):
            loss_total[i]=loss_1[i] 
    
    ml1=loss_total<=laux
    return ml1, l
   

def random_trans_combo(tensor, df=False):
    # tensor: bs * c * h * w
    if not df:
        tensor += (torch.randn_like(tensor)*0.1).clamp(0,1)
    if torch.rand(1) > 0.5 or df:
        tensor = tensor.flip(3)
    if not df:
        r_h = torch.randint(0, 8, (1,)).item()
        # print('r_h:{}'.format(r_h))
        r_w = torch.randint(0, 8, (1,)).item()
        h = torch.randint(24, int(32-r_h), (1,))
        w = torch.randint(24, int(32-r_w), (1,))
    else:
        r_h, r_w, h, w = 2, 2, 28, 28

    tensor = tensor[:, :, int(r_h):int(r_h+h), int(r_w):int(r_w+w)]
    return nn.functional.interpolate(tensor, [32, 32])

def random_trans_combo_tiny(tensor, df=False):
    # tensor: bs * c * h * w
    if not df:
        tensor += (torch.randn_like(tensor)*0.1).clamp(0,1)
    if torch.rand(1) > 0.5 or df:
        tensor = tensor.flip(3)
    if not df:
        r_h = torch.randint(0, 16, (1,)).item()
        # print('r_h:{}'.format(r_h))
        r_w = torch.randint(0, 16, (1,)).item()
        h = torch.randint(48, int(64-r_h), (1,))
        w = torch.randint(48, int(64-r_w), (1,))
    else:
        r_h, r_w, h, w = 4, 4, 56, 56

    tensor = tensor[:, :, int(r_h):int(r_h+h), int(r_w):int(r_w+w)]
    return nn.functional.interpolate(tensor, [64, 64])

def rotate_images(X, degree=0):
    if degree == 0:
        return X
    elif degree == 90:
        return X.transpose(2,3).flip(3)
    elif degree == 180:
        return X.flip(2).flip(3)
    elif degree == 270:
        return X.transpose(2,3).flip(2)

def ent_num_batch(pred_pfy,ent):
    loss_ent = pred_pfy.softmax(1)
    num_class=pred_pfy.shape[1]
    m = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1).mean(0)/math.log(num_class)
    ml2=m<=ent
    return ml2

def ent_num(pred_pfy,ent):
    loss_ent = pred_pfy.softmax(1)
    num_class=pred_pfy.shape[1]
    m = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1)/math.log(num_class)
    ml2=m<=ent
    return ml2

def pfy(ml3,X_pfy_denoise,X_pfy_denoise_last):
    for i,iterm in enumerate(ml3):
        if iterm:
            X_pfy_denoise_last[i]=X_pfy_denoise[i]


    return X_pfy_denoise_last





