import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from resnet_ae import *
from resnet import *
from wide_resnet import *
from wideresnet_ae import *
import os

def load_model(name, dataset, n_class=10, in_channel=3, save_dir=None, substitute=False):
    if name == 'fcnet':
        model = FCNet(n_class=n_class, in_dim=784, hidden_dim=128)
    elif name == 'cnet':
        model = CNet(n_class=n_class, in_channel=in_channel)
    elif name == 'ae':
        model = AutoEncoder(n_class=n_class, in_dim=784, hidden_dim=128)
    elif name == 'cae':
        model = ConvAutoEncoder(n_class=n_class, in_channel=in_channel)
    elif name == 'cae_cifar':
        model = AE_VGG(3)
        print('aevgg')
    elif name == 'resnetae_cifar':
        model = resnet_AE(z_dim=512,n_class=n_class)
        print('resnetae_cifar')
    elif name == 'resnet_14ae_cifar':
        model = resnet_AE_14(z_dim=64,n_class=n_class)
        print('resnet_14ae_cifar')
    elif name == 'resnet':
        model = ResNet_(18, n_class)
    elif name == 'wide-resnet':
        model = Wide_ResNet_(28, 10, 0.3, n_class)
    elif name == 'wide-resnet_ae':
        model = wideresnet_AE(z_dim=640,n_class=n_class)
        print('wide-resnet_ae_cifar')
    elif name == 'resnet-rot':
        model = ResNet(n_class=n_class)
    elif name == 'wide-resnet-rot':
        model = WResNet(n_class=n_class)
    else:
        raise TypeError("Unrecognized model name: {}".format(name))

    if dataset == 'cifar10':
        model.add_normalizer(normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
    elif dataset == 'cifar100':
        model.add_normalizer(normalizer(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]))

    if save_dir is not None:
        print('load')
        if substitute:
            # subpath = './results/mnist/cae_dr_aux'
            if dataset == 'cifar10':
                subpath = './results/cifar10/resnet14_ae_aux'
                # subpath ='./results/cifar10/widresnet/normal'
                # subpath = './results/cifar10/resnet/normal'
            # subpath = './results/cifar100/widresnet/normal'
            # subpath = './results/cifar100/wide-resnet_ae_noise'
            # model.load_state_dict(torch.load(os.path.join(subpath, 'latest_model.pth'), map_location='cpu'))
            model.load_state_dict(torch.load(os.path.join(save_dir, 'substitute_{}.pth'.format(name)), map_location='cpu'))
        else:
            print("load model yes")
            model.load_state_dict(torch.load(os.path.join(save_dir, 'latest_model.pth'), map_location='cpu'))
            # model.load_state_dict(torch.load(os.path.join(save_dir, 'latest_model.pth'), map_location='cpu'))
    return model


class normalizer(nn.Module):

    def __init__(self, mean, std):
        super(normalizer, self).__init__()
        self.mean = torch.FloatTensor(mean)[:, None, None]
        self.std = torch.FloatTensor(std)[:, None, None]

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class add_noise(nn.Module):

    def __init__(self, std):
        super(add_noise, self).__init__()
        self.std = std

    def forward(self, x):
        return (x + torch.randn_like(x)*self.std).clamp(0,1)


class FCNet(nn.Module):

    def __init__(self, n_class, in_dim, hidden_dim=128, nonlinear='Relu'):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_class)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, return_reps=False):

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.cls(x)
        return x


class CNet(nn.Module):
    def __init__(self, n_class, in_channel=3, hidden_dim=1024):

        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7*7*64, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_class)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, return_reps=False):

        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 7*7*64)
        x = self.relu(self.fc1(x))
        x = self.cls(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, n_class, in_dim, hidden_dim=128):

        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, in_dim)
        self.cls = nn.Linear(hidden_dim, n_class)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        if add_noise:
            x = (x + torch.randn_like(x)*0.5).clamp(0,1)
        size = x.shape
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if return_reps:
            return x


        l = self.cls(x)
        self.pred = l
        
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        self.r = x.reshape(*size)
        if self.aux:
            return self.r

        return l


class ConvAutoEncoder(nn.Module):

    def __init__(self, n_class, in_channel=3, hidden_dim=1024, out_channel=None):

        super(ConvAutoEncoder, self).__init__()
        if not out_channel:
            out_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.fc1 = nn.Linear(7*7*64, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_class)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc2 = nn.Linear(hidden_dim, 7*7*64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(32, out_channel, 3, stride=2, padding=1, output_padding=1)

        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        size = x.shape
        # print("size:{}".format(size))
        if add_noise:
            # print('add noise')
            x = (x + torch.randn_like(x)*0.5).clamp(0,1)

        # x = self.normalizer(x)
        x = self.relu(self.conv1(x))
        # print("conv1.shape:{}".format(x.shape))
        x = self.relu(self.conv2(x))
        # print("conv2.shape:{}".format(x.shape))

        x = x.view(-1, 7*7*64)

        x = self.relu(self.fc1(x))
        # print("relu.shape:{}".format(x.shape))

        if return_reps:
            return x

        l = self.cls(x)
        # print("cls.shape:{}".format(l.shape))
        self.pred = l

        x = self.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))

        self.r = x.reshape(*size)
        if self.aux:
            return self.r
        return l


    def add_normalizer(self, normalizer):
        self.normalizer = normalizer



vgg16_layers = {
                0:[64, 64, 'M'],
                1:[128, 128, 'M'],
                2:[256, 256, 256, 'M'],
                3:[512, 512, 512, 'M'],
                4:[512, 512, 512, 'M']
               }

class AutoEncoder_VGG(nn.Module):
    def __init__(self, total_layers):
        super(AutoEncoder_VGG, self).__init__()
        self.encoder = self.make_encoder(total_layers, vgg16_layers)
        self.decoder = self.make_decoder(total_layers)
    def make_encoder(self, total_layers, vgg_layers):
        layers = []
        in_channels = 3
        for i in range(total_layers):
            for l in vgg_layers[i]:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
                else:
                    layers += [
                        nn.Conv2d(in_channels, l, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(l),
                        nn.ReLU(inplace=True)
                    ]
                    in_channels = l
        return nn.Sequential(*layers)
    def make_decoder(self, total_layers):
        layers = []
        in_channel = int(64*(2**(total_layers-1)))
        out_channel = in_channel//2
        for i in range(total_layers):
            if i == (total_layers -1):
                out_channel = 3
            layers += [
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel//2
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class VGG16_classifier(nn.Module):
    def __init__(self, total_layers):
        super(VGG16_classifier, self).__init__()
        self.classifier = self.make_fc(total_layers)
        self.total_layers = total_layers
    def make_fc(self, total_layers):
        layers = []
        if total_layers == 1:
            layers += [
                nn.Linear(16384, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 2:
            layers += [
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 3:
            layers += [
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 4:
            layers += [nn.Linear(2048,10)]
        elif total_layers == 5:
            layers += [nn.Linear(512,10)]
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG16(nn.Module):
    def __init__(self, in_channels, total_layers):
        super(VGG16, self).__init__()
        self.features = self.make_feature_layers(in_channels, total_layers, vgg16_layers)
        self.classifier = nn.Linear(vgg16_layers[total_layers-1][0], 10)
    def make_feature_layers(self, in_channels, total_layers, vgg_layers):
        layers = []
        for i in range(total_layers):
            for l in vgg_layers[i]:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
                else:
                    layers += [
                        nn.Conv2d(in_channels, l, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(l),
                        nn.ReLU(inplace=True)
                    ]
                    in_channels = l
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AE_VGG(nn.Module):
    def __init__(self, autoencoder_layers):
        super(AE_VGG, self).__init__()
        self.autoencoder = AutoEncoder_VGG(autoencoder_layers)
        self.vgg16 =VGG16_classifier(autoencoder_layers)
        # self.vgg16 = VGG16(int(64 * (2 ** (autoencoder_layers - 1))), 5 - autoencoder_layers)
        self.ae_layers = autoencoder_layers
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):
        size = x.shape
        # print("size:{}".format(size))
        if add_noise:
            # print('add noise')
            x = (x + torch.randn_like(x) * 0.1).clamp(0, 1)

        x = self.normalizer(x)

        encoded, decoded = self.autoencoder(x)
        self.r = decoded.reshape(*size)

        if return_reps:
            return encoded

        l = self.vgg16(encoded)
        self.pred = l
        if self.aux:
            return self.r
        return l

        # return encoded, decoded, classification

    def add_normalizer(self, normalizer):
        self.normalizer = normalizer

class resnet_AE(nn.Module):
    def __init__(self, z_dim,n_class):
        super(resnet_AE, self).__init__()

        self.encoder = ResNet18Enc(hiddendim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)
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

class FCNet_rotate(nn.Module):

    def __init__(self, n_class, in_dim, hidden_dim=128):

        super(FCNet_rotate, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, n_class)
        self.fc4 = nn.Linear(32, 4)
        self.relu = nn.ReLU(inplace=True)
        self.aux = False

    def forward(self, x):

        size = x.shape
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        self.pred_deg = self.fc4(x)
        if self.aux:
            return self.pred_deg
        return self.fc3(x)


class ResNet(nn.Module):

    def __init__(self, n_class):

        super(ResNet, self).__init__()
        self.resnet = ResNet_(18, n_class)
        self.fc1 = nn.Linear(64, 4)
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        if add_noise:
            x = (x + torch.randn_like(x)*0.1).clamp(0,1)
        l = self.resnet(x)

        if return_reps:
            return self.resnet.x
        self.pred_deg = self.fc1(self.resnet.x)
        if self.aux:
            return self.pred_deg
        self.pred = l
        return l

    def add_normalizer(self, normalizer):
        self.resnet.add_normalizer(normalizer) 


class WResNet(nn.Module):

    def __init__(self, n_class, k=10):

        super(WResNet, self).__init__()
        self.resnet = Wide_ResNet_(28, k, 0.3, n_class)
        self.fc1 = nn.Linear(k*64, 4)
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        if add_noise:
            x = (x + torch.randn_like(x)*0.1).clamp(0,1)
        # normalization in wide-resnet
        l = self.resnet(x)

        if return_reps:
            return self.resnet.x
            
        self.pred_deg = self.fc1(self.resnet.x)
        if self.aux:
            return self.pred_deg
        self.pred = l
        return l

    def add_normalizer(self, normalizer):
        self.resnet.add_normalizer(normalizer) 
