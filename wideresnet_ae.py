import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision
from wide_resnet import *

#用上采样加卷积代替了反卷积
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class Wide_ResNet_encoder(nn.Module):
    def __init__(self, hiddendim):
        super(Wide_ResNet_encoder, self).__init__()
        self.hiddendim = hiddendim
        self.Wide_ResNet =  Wide_ResNet_28_10(28, 10, 0.3)
        # self.num_feature = self.ResNet18.fc.in_features
        # self.ResNet18.fc = nn.Linear(self.num_feature, self.z_dim)

    def forward(self, x):
        x_rep = self.Wide_ResNet(x)
        # mu = x[:, :self.z_dim]
        # logvar = x[:, self.z_dim:]
        return x_rep

    def add_normalizer(self, normalizer):
        self.normalizer = normalizer

class Wide_ResNet_28_10(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate):
        super(Wide_ResNet_28_10, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        # self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out) # 160 * 32 * 32
        # self.x = out.mean(dim=(2,3))
        out = self.layer2(out) # 320 * 16 * 16
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        self.x = out
        # out = self.linear(out)
        #
        # self.pred = out
        return out



class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockDec, self).__init__()

        # planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride != 1 or in_planes != planes:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()


    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class wideResNetDec(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate,nc=3):
        super().__init__()
        self.in_planes = 640
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.layer3 = self._make_layer(BasicBlockDec, nStages[3], n,outnstage=nStages[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, nStages[2], n, outnstage=nStages[1],stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, nStages[1], n,outnstage=nStages[0], stride=1)
        # self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(nStages[0], nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, block, planes, num_blocks,outnstage, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        # print(strides)
        i=0
        for stride in reversed(strides):
            if i==3:
                planes=outnstage
            layers.append(block(self.in_planes, planes, stride))
            # print(layers)
            self.in_planes = planes
            i+=1
        return nn.Sequential(*layers)

    def forward(self, z):
        # x = self.linear(z)

        x = z.view(z.size(0),640, 1, 1)
        # print(x.shape)
        x = F.interpolate(x, scale_factor=8)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        # x = self.layer1(x)
        # x = F.interpolate(x, size=(32, 32), mode='bilinear')
        x = torch.sigmoid(self.conv1(x))
        # print(x.shape)
        x = x.view(x.size(0), 3, 32, 32)
        # print(x.shape)
        return x

class wideresnet_AE(nn.Module):
    def __init__(self, z_dim,n_class):
        super(wideresnet_AE, self).__init__()

        self.encoder = Wide_ResNet_encoder(hiddendim=z_dim)
        self.decoder = wideResNetDec(28, 10, 0.3,nc=3)
        # self.fc1=nn.Linear(z_dim,128)
        self.cls = nn.Linear(z_dim, n_class)
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):
        size = x.shape
        # print("size:{}".format(size))
        if add_noise:
            # print('add noise')
            x = (x + torch.randn_like(x) * 0.1).clamp(0, 1)

        x = self.normalizer(x)
        encoder = self.encoder(x)
        # print(encoder.shape)

        decoder = self.decoder(encoder)
        self.r = decoder.reshape(*size)
        if self.aux:
            return self.r

        # output = self.relu(self.fc1(encoder))
        l = self.cls(encoder)
        # print("cls.shape:{}".format(l.shape))
        self.pred = l
        return l


    def add_normalizer(self, normalizer):
        self.normalizer = normalizer




class normalizer(nn.Module):

    def __init__(self, mean, std):
        super(normalizer, self).__init__()
        self.mean = torch.FloatTensor(mean)[:, None, None]
        self.std = torch.FloatTensor(std)[:, None, None]

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)





if __name__ == '__main__':
    model = wideresnet_AE(z_dim=640,n_class=10)
    model.add_normalizer(normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
    # print(model)
    input = torch.randn(1, 3, 32, 32)
    out = model(input)
    # print(out.shape)

#
# vae = VAE(z_dim=256).cuda()
# optimizer = optim.Adam(vae.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#
# root = "./dataset/"
#
#
#
# transform = transforms.Compose([transforms.Resize([224, 224]),
#                                 transforms.ToTensor(),
#                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
#                                 # gray -> GRB 3 channel (lambda function)
#                                 transforms.Normalize(mean=[0.0, 0.0, 0.0],
#                                                      std=[1.0, 1.0, 1.0])])  # for grayscale images
#
# # MNIST dataset (images and labels)
# MNIST_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# MNIST_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
#
# # Data loader (input pipeline)
# train_iter = torch.utils.data.DataLoader(dataset=MNIST_train_dataset, batch_size=batch_size, shuffle=True)
# test_iter = torch.utils.data.DataLoader(dataset=MNIST_test_dataset, batch_size=batch_size, shuffle=False)
#
# for epoch in range(0, epoch_num):
#     l_sum = 0
#     scheduler.step()
#     for x, y in train_iter:
#         # x = torch.sigmoid(x).cuda()
#         x = x.cuda()
#         print(x.requires_grad)
#         optimizer.zero_grad()
#         recon_x, mu, logvar = vae.forward(x)
#         loss = loss_func(recon_x, x, mu, logvar)
#         l_sum += loss
#         loss.backward()
#         optimizer.step()
#     print("loss\n", l_sum)
#     print(epoch, "\n")
#
# i = 0
# with torch.no_grad():
#     for t_img, y in test_iter:
#         t_img = Variable(t_img).cuda()
#         result, mu, logvar = vae.forward(t_img)
#         utils.save_image(result.data, str(i) + '.png', normalize=True)
#         i += 1
