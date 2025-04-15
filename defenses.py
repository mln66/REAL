import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial

from attacks import fgsm, pgd_linf, inject_noise,pgd_linf_1

def defense_wrapper(model, criterion, X, defense, epsilon=None, step_size=None, num_iter=None):
    
    # model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta = pgd_linf(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon, step_size=step_size, num_iter=num_iter)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    else:
        raise TypeError("Unrecognized defense name: {}".format(defense))
    model.aux = False
    # model.eval()
    return inv_delta

def purify(model, aux_criterion, X, defense_mode='pgd_linf', delta=4/255, step_size=4/255, num_iter=5):

    if aux_criterion is None:
        return X
    aux_track = torch.zeros(11, X.shape[0])
    inv_track = torch.zeros(11, *X.shape)
    for e in range(11):
        # print("e:{}".format(e))
        defense = partial(defense_wrapper, criterion=aux_criterion, defense=defense_mode, epsilon=e*delta, step_size=step_size, num_iter=num_iter)
        inv_delta = defense(model, X=X)
        inv_track[e] = inv_delta
        # print("aux_track")
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1)).detach()
    e_selected = aux_track.argmin(dim=0)
    return inv_track[e_selected, torch.arange(X.shape[0])].to(X.device) + X





def defense_wrapper_1(model, criterion, X,X_denoise, defense, epsilon=None, step_size=None, num_iter=None):
    # model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X,X_denoise: -criterion(model, X,X_denoise), X=X,X_denoise=X_denoise, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta = pgd_linf_1(model, lambda model, X, X_denoise: -criterion(model, X, X_denoise), X=X,
                                 X_denoise=X_denoise, epsilon=epsilon, step_size=step_size, num_iter=num_iter)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    else:
        raise TypeError("Unrecognized defense name: {}".format(defense))
    model.aux = False
    # model.eval()
    return inv_delta

def purify_1(model, aux_criterion, X,X_denoise, defense_mode='pgd_linf', delta=4/255, step_size=4/255, num_iter=5):

    if aux_criterion is None:
        return X_denoise
    aux_track = torch.zeros(7, X_denoise.shape[0])
    inv_track = torch.zeros(7, *X_denoise.shape)
    for e in range(7):
        # print("e:{}".format(e))
        defense = partial(defense_wrapper_1, criterion=aux_criterion, defense=defense_mode, epsilon=e*delta, step_size=step_size, num_iter=num_iter)
        inv_delta = defense(model, X=X,X_denoise=X_denoise)
        inv_track[e] = inv_delta
        # print("aux_track")
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1),((X_denoise+inv_delta).clamp(0,1)),test=True).detach()
    e_selected = aux_track.argmin(dim=0)
    # print('e_select:{}'.format(e_selected))
    return inv_track[e_selected, torch.arange(X_denoise.shape[0])].to(X_denoise.device) + X_denoise


def defense_wrapper_2(model, criterion, X, defense,epsilon=None, step_size=None, num_iter=None):
    # model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta = pgd_linf(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon, step_size=step_size,num_iter=num_iter)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    else:
        raise TypeError("Unrecognized defense name: {}".format(defense))
    model.aux = False
    # model.eval()
    return inv_delta


def purify_2(model, aux_criterion, X, defense_mode='pgd_linf', delta=4/255, step_size=4/255, num_iter=5):

    if aux_criterion is None:
        return X
    aux_track = torch.zeros(7, X.shape[0])
    inv_track = torch.zeros(7, *X.shape)
    for e in range(7):
        # print("e:{}".format(e))
        defense = partial(defense_wrapper_2, criterion=aux_criterion, defense=defense_mode, epsilon=e*delta, step_size=step_size, num_iter=num_iter)
        inv_delta = defense(model, X=X)
        inv_track[e] = inv_delta
        # print("aux_track")
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1),test=True).detach()
    e_selected = aux_track.argmin(dim=0)
    return inv_track[e_selected, torch.arange(X.shape[0])].to(X.device) + X
