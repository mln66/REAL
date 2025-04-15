import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(0)

def joint_criterion(model, aux_criterion, X, y, alpha=1.0):
    aux_loss, l = aux_criterion(model, X, joint=True, train=True)
    # print("l.shape:{}".format(l.shape))
    # print("auxloss{}".format(aux_loss))
    # print("l{}".format(l))
    if aux_criterion.__name__ == 'recon_criterion':
        y = y
    elif aux_criterion.__name__ == 'pi_criterion':
        y = y.repeat(2)
    elif aux_criterion.__name__ == 'rotate_criterion':
        y = y.repeat(4)
    loss = nn.functional.cross_entropy(l, y)
    
    return loss + aux_loss * alpha, (loss, aux_loss)

#l means prediction
def shot(l):

    loss_ent = l.softmax(1)
    loss_ent = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1).mean(0)
    loss_div = l.softmax(1).mean(0)
    loss_div = torch.sum(loss_div * torch.log(loss_div + 1e-5))
    # return loss_ent + loss_div
    return loss_ent

#l means prediction
def shot_X(l,X):

    loss_ent1 = torch.zeros_like(l)
    loss_ent = l.softmax(1)


    # 将除第一个维度之外的所有维度展平为第二维
    X_flatten = X.view(X.size(0), -1)
    non_zero_mask = ~torch.all(X_flatten == 0, dim=1)

    if loss_ent.shape[0]!=X_flatten.shape[0]:
        shape_x=X_flatten.shape[0]
        loss_ent1[:shape_x][non_zero_mask] =loss_ent[:shape_x][non_zero_mask]

    if not torch.any(non_zero_mask):  # 检查是否存在非零样本
        return 0

    num = non_zero_mask.sum()  # 非零样本的数量
    loss_ent = -torch.sum( torch.sum(loss_ent1 * torch.log(loss_ent1 + 1e-5), dim=1), dim=0 ) / num
    return loss_ent


def joint_tent_rec(model,X,X_denoise,test=False):

    if test:
        reduction = 'elementwise_mean'
    else:
        reduction = 'sum'
    # reduction = 'mean'
    aux_loss, l = recon_criterion_1(model, X.clamp(0,1),X_denoise.clamp(0,1), joint=True, train=False,reduction=reduction)
    # print('l.shape:{}'.format(l.shape))
    loss_ent=shot_X(l,X)
    # loss_ent=shot(l)
    num_class=l.shape[1]
    loss_ent=loss_ent/math.log(num_class)
    if test:
        purify_p=1
    else:
        purify_p = math.pow(loss_ent, 2)

    loss = aux_loss + 0.25*purify_p*loss_ent
    return loss


def joint_futent_rec(model,X,X_denoise,test=False):
    if test:
        reduction = 'elementwise_mean'
    else:
        reduction = 'sum'
    # reduction = 'mean'
    aux_loss, l = recon_criterion_1(model, X,X_denoise, joint=True, train=False,reduction=reduction)
    loss_ent=shot_X(l,X)

    num_class=l.shape[1]
    loss_ent=loss_ent/math.log(num_class)
    if test:
        purify_p=1
    else:
        purify_p = math.pow(1 - loss_ent, 2)

    loss = aux_loss -0.25* purify_p * loss_ent

    return loss


def joint_tent_LC(model,X,test=False):
    # print('X.shape:{}'.format(X.shape))
    aux_loss, l = pi_criterion(model, X, joint=True, train=False)
    # loss_ent=shot(l)
    loss_ent=shot_X(l,X)
    num_class=l.shape[1]
    loss_ent=loss_ent/math.log(num_class)
    
    # print(loss_ent)
    # attack_p = 2 * (1 / (1 + math.exp(-loss_ent)) - 1 / 2)
    if test:
        purify_p=1
    else:
        purify_p = math.pow(loss_ent, 2)

    # purify_p=1
    loss = aux_loss + 0.25* purify_p* loss_ent

    return loss

def joint_futent_LC(model,X,test=False):
    # print('X.shape:{}'.format(X.shape))
    aux_loss, l = pi_criterion(model, X, joint=True, train=False)
    # loss_ent=shot(l)
    loss_ent=shot_X(l,X)
    num_class=l.shape[1]
    loss_ent=loss_ent/math.log(num_class)
    # print(loss_ent)
    
    # attack_p = 2 * (1 / (1 + math.exp(-loss_ent)) - 1 / 2)
    if test:
        purify_p=1
    else:
        purify_p = math.pow(1 - loss_ent, 2)
    # print('purify_p{}'.format(purify_p))
    # purify_p = 1
    loss = aux_loss - 0.25* purify_p * loss_ent
    # loss = aux_loss -0.25*a*loss_ent
    return loss




def recon_criterion(model, X, joint=False, train=False, reduction='mean'):
    l = model(X, add_noise=train)
    # print('recon_l.shape:{}'.format(l.shape))
    loss = nn.functional.mse_loss(model.r, X,reduction='elementwise_mean')
    # print('122')
    # print("aux_loss:{}".format(loss))
    if not joint:
        return loss
    return loss, l

def recon_criterion_1(model, X,X_denoise, joint=False, train=False, reduction='elementwise_mean'):
    # print(X.shape)
    # print(X_denoise.shape)
    l = model(X_denoise, add_noise=train) #
    # print('recon_l.shape:{}'.format(l.shape))

    loss = nn.functional.mse_loss(model.r, X,reduction=reduction)
    # print('123')
    # print("aux_loss:{}".format(loss))
    if not joint:
        return loss
    return loss, l


def pi_criterion(model, X,joint=False, train=False, reduction='mean'):     
    if not train:
        X1 = X
    else:
        if X.shape[-1]==64:
            X1 = random_trans_combo_tiny(X)
        else:
            X1 = random_trans_combo(X)
    if X.shape[-1]==64:
        X2 = random_trans_combo_tiny(X, df=~train)
    else:
        X2 = random_trans_combo(X, df=~train)
  
    l1= model(X1)
  
    l2=model(X2)
    l = torch.cat((l1, l2), dim=0)
    loss = nn.functional.mse_loss(l1, l2,reduction=reduction)
    if not joint:
        return loss
    return loss, l



            
                    
def pi_criterion_cln(model, X,m11,joint=False, train=False, reduction='mean'):
    if not train:
        X1 = X
    else:
        if X.shape[-1]==64:
            X1 = random_trans_combo_tiny(X)
        else:
            X1 = random_trans_combo(X)
    if X.shape[-1]==64:
        X2 = random_trans_combo_tiny(X, df=~train)
    else:
        X2 = random_trans_combo(X, df=~train)
    l1, l2 = model(X1), model(X2)
    l = torch.cat((l1, l2), dim=0)
    loss=0
    j=0
    for i, item in enumerate(m11):
        if item:
            loss_1=nn.functional.mse_loss(l1[i], l2[i],reduction=reduction)
            # print('l1[i]:{}'.format(l1[i]))
            # print('loss_1:{}'.format(loss_1))
            loss+=loss_1
            j+=1
    return loss, j

def rec_criterion_cln(model, X,X_denoise,m11,joint=False, train=False, reduction='mean'):
    l = model(X_denoise, add_noise=train)  #
    # print('recon_l.shape:{}'.format(l.shape))
    loss=0
    j=0
    for i, item in enumerate(m11):
        if item:
            loss_1=nn.functional.mse_loss(model.r[i], X[i])
            print('X[i]:{}'.format(X[i].shape))
            print('loss_1:{}'.format(loss_1))
            loss+=loss_1
            j+=1
    return loss, j






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

# def second_order(model, X, y, df_criterion, beta=1):
#     pred = model(X)
#     lent=shot(pred)
#     return nn.functional.cross_entropy(pred, y) - df_criterion(model, X) * 1+lent*beta
def second_order(model, X, y, df_criterion, beta=1):
    pred = model(X)
    lent = shot(pred)
    num_class=pred.shape[1]
    lent=lent/math.log(num_class)
    return nn.functional.cross_entropy(pred, y) - df_criterion(model, X) * 1+ lent*beta

# def second_order(model, X, y, df_criterion, beta=1):
#     pred = model(X)
#
#     return nn.functional.cross_entropy(pred, y) - df_criterion(model, X) * beta * 100