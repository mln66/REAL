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
from resnet import *

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



class ResNet18Enc(nn.Module):
    def __init__(self, hiddendim,n_class):
        super(ResNet18Enc, self).__init__()
        self.hiddendim = hiddendim
        self.ResNet18 =  ResNet_11(ResidualBlock,num_classes=n_class)
        # self.num_feature = self.ResNet18.fc.in_features
        # self.ResNet18.fc = nn.Linear(self.num_feature, self.z_dim)

    def forward(self, x):
        x = self.ResNet18(x)
        # mu = x[:, :self.z_dim]
        # logvar = x[:, self.z_dim:]
        return x

class ResNet18Enc_14(nn.Module):
    def __init__(self, hiddendim,n_class):
        super(ResNet18Enc_14, self).__init__()
        self.hiddendim = hiddendim
        self.ResNet18 =  ResNet_14(18,n_class)
        # self.num_feature = self.ResNet18.fc.in_features
        # self.ResNet18.fc = nn.Linear(self.num_feature, self.z_dim)

    def forward(self, x):
        x = self.ResNet18(x)
        # mu = x[:, :self.z_dim]
        # logvar = x[:, self.z_dim:]
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, n_class=10,nc=3):
        super().__init__()
        self.in_planes = 512

        # self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        # x = self.linear(z)
        x = z.view(z.size(0), 512, 1, 1)
        # print(x.shape)
        x = F.interpolate(x, scale_factor=4)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        # x = F.interpolate(x, size=(32, 32), mode='bilinear')
        x = torch.sigmoid(self.conv1(x))
        # print(x.shape)
        x = x.view(x.size(0), 3, 32, 32)
        # print(x.shape)
        return x

class ResNet18Dec_14(nn.Module):

    def __init__(self,  z_dim=32, n_class=10,nc=3):
        super().__init__()
        self.in_planes = 64

        # self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 16, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 16, 2, stride=1)
        # self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(16, nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        # x = self.linear(z)

        x = z.view(z.size(0),64, 1, 1)
        # print(x.shape)
        x = F.interpolate(x, scale_factor=8)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        # x = self.layer1(x)
        # x = F.interpolate(x, size=(32, 32), mode='bilinear')
        x = torch.sigmoid(self.conv1(x))
        # print(x.shape)
        x = x.view(x.size(0), 3, 32, 32)
        # print(x.shape)
        return x

class normalizer(nn.Module):

    def __init__(self, mean, std):
        super(normalizer, self).__init__()
        self.mean = torch.FloatTensor(mean)[:, None, None]
        self.std = torch.FloatTensor(std)[:, None, None]

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class resnet_AE(nn.Module):
    def __init__(self, z_dim,n_class):
        super(resnet_AE, self).__init__()

        self.encoder = ResNet18Enc(hiddendim=z_dim,n_class=n_class)
        self.decoder = ResNet18Dec(z_dim=z_dim,n_class=n_class)
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

        l = self.cls(encoder)
        # print("cls.shape:{}".format(l.shape))
        self.pred = l
        return l

    def add_normalizer(self, normalizer):
        self.normalizer = normalizer

    # @staticmethod
    # def reparameterize(mean, logvar):
    #     std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
    #     epsilon = torch.randn_like(std).cuda()
    #     return epsilon * std + mean

class resnet_AE_14(nn.Module):
    def __init__(self, z_dim,n_class):
        super(resnet_AE_14, self).__init__()

        self.encoder = ResNet18Enc_14(hiddendim=z_dim,n_class=n_class)
        self.decoder = ResNet18Dec_14(z_dim=z_dim,n_class=n_class)
        self.cls = nn.Linear(z_dim, n_class)
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):
        size = x.shape
        # print("size:{}".format(size))
        # print(add_noise)
        # if add_noise:
        #     # print('add noise')
        #     x = (x + torch.randn_like(x) * 0.1).clamp(0, 1)
        # print(add_noise)
        x = self.normalizer(x)
        encoder = self.encoder(x)
        # print(encoder.shape)

        decoder = self.decoder(encoder)
        self.r = decoder.reshape(*size)
        if self.aux:
            return self.r

        l = self.cls(encoder)
        # print("cls.shape:{}".format(l.shape))
        self.pred = l
        return l

    def add_normalizer(self, normalizer):
        self.normalizer = normalizer




class ResNet_11(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet_11, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        # print(strides)
        layers = []
        for stride in strides:
            # print(stride)
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        # out = self.fc(out)
        return out

class ResNet_14(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet_14, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.normalizer(x)
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        # self.x = out.view(out.size(0), -1)
        out = self.layer3(out)
        # print(out.shape)
        l=out.shape[-1]
        out = F.avg_pool2d(out, l)
        # print('out.size:{}'.format(out.shape))
        out = out.view(out.size(0), -1)
        # print('out.size:{}'.format(out.shape))
        self.x = out


        # out = self.linear(out)
        #
        # self.pred = out
        return out



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# def ResNet18():
#     return ResNet_11(ResidualBlock)



if __name__ == '__main__':
    model = resnet_AE_14(z_dim=64,n_class=10)
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
