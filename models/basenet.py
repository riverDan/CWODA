import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torch.autograd.variable import *
from efficientnet_pytorch import EfficientNet
import math


def get_input_channels(args=None):
    feature = str(getattr(args, 'feature', '')).lower().replace(',', '+').replace('_', '+')
    return 6 if feature in {'stft+dscg', 'dscg+stft'} else 3


def repeat_conv_weight(weight, in_channels):
    if weight.size(1) == in_channels:
        return weight
    repeat_count = int(math.ceil(in_channels / float(weight.size(1))))
    repeated = weight.repeat(1, repeat_count, 1, 1)[:, :in_channels, :, :]
    return repeated * (weight.size(1) / float(in_channels))


def adapt_conv2d_input_channels(conv, in_channels):
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(repeat_conv_weight(conv.weight, in_channels))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass


## Some classes from https://github.com/ksaito-ut/OPDA_BP/blob/master/models/basenet.py

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50', model_path=None, normalize=True,
                 return_reconstruction=False, input_channels=3):
        super(ResNetFc, self).__init__()

        self.model_resnet = resnet_dict[model_name](pretrained=True)
        self.return_reconstruction = return_reconstruction

        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.input_channels = input_channels
        self.conv1 = adapt_conv2d_input_channels(model_resnet.conv1, input_channels)
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        # del self.mo
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

        self.decoder = None
        if self.return_reconstruction:
            self.decoder = nn.Sequential(
                nn.Linear(self.__in_features, 512 * 7 * 7),
                nn.ReLU(True),
                nn.Unflatten(1, (512, 7, 7)),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Sigmoid())

    def get_mean(self, device):
        if self.mean is False or self.mean.device != device:
            base_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            mean = np.tile(base_mean, int(math.ceil(self.input_channels / 3.0)))[:self.input_channels]
            self.mean = torch.from_numpy(mean.reshape((1, self.input_channels, 1, 1))).to(device)
        return self.mean

    def get_std(self, device):
        if self.std is False or self.std.device != device:
            base_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            std = np.tile(base_std, int(math.ceil(self.input_channels / 3.0)))[:self.input_channels]
            self.std = torch.from_numpy(std.reshape((1, self.input_channels, 1, 1))).to(device)
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean(x.device)) / self.get_std(x.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x = F.dropout(x, p=0.5, training=self.training)

        recon = self.decoder(x) if self.decoder is not None else None

        return x, recon

    def output_num(self):
        return self.__in_features


# class ResNetFc(BaseFeatureExtractor):
#     def __init__(self, model_name='resnet50', model_path=None, normalize=True):
#         super(ResNetFc, self).__init__()

#         self.model_resnet = resnet_dict[model_name](pretrained=True)

#         if model_path:
#             self.model_resnet.load_state_dict(torch.load(model_path))
#         if model_path or normalize:
#             self.normalize = True
#             self.mean = False
#             self.std = False
#         else:
#             self.normalize = False

#         model_resnet = self.model_resnet
#         self.conv1 = model_resnet.conv1
#         self.bn1 = model_resnet.bn1
#         self.relu = model_resnet.relu
#         self.maxpool = model_resnet.maxpool
#         self.layer1 = model_resnet.layer1
#         self.layer2 = model_resnet.layer2
#         self.layer3 = model_resnet.layer3
#         self.layer4 = model_resnet.layer4

#         # del self.mo
#         self.avgpool = model_resnet.avgpool
#         self.__in_features = model_resnet.fc.in_features

#         # 为 x_seq 添加专门的 conv1 层（用于单通道输入）
#         self.seq_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # 初始化为原始 conv1 的平均（适应预训练权重）
#         with torch.no_grad():
#             self.seq_conv1.weight.copy_(self.conv1.weight.mean(dim=1, keepdim=True))

#         # 特征融合投影层
#         self.img_proj = nn.Linear(512, 256)  # 原始是2048维
#         self.seq_proj = nn.Linear(512, 256)


#     def get_mean(self):
#         if self.mean is False:
#             self.mean = Variable(
#                 torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
#         return self.mean

#     def get_std(self):
#         if self.std is False:
#             self.std = Variable(
#                 torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
#         return self.std

#     def forward(self, x, x_seq):
#         if self.normalize:
#             x = (x - self.get_mean()) / self.get_std()
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)


#         # x_seq: 单通道序列图像特征提取
#         x_seq = self.seq_conv1(x_seq)
#         x_seq = self.bn1(x_seq)
#         x_seq = self.relu(x_seq)
#         x_seq = self.maxpool(x_seq)
#         x_seq = self.layer1(x_seq)
#         x_seq = self.layer2(x_seq)
#         x_seq = self.layer3(x_seq)
#         x_seq = self.layer4(x_seq)
#         x_seq = self.avgpool(x_seq)
#         x_seq = x_seq.view(x_seq.size(0), -1)

#         img_proj = self.img_proj(x)
#         seq_proj = self.seq_proj(x_seq)
#         fused_features = torch.cat([img_proj, seq_proj], dim=1)

#         return fused_features

#     def output_num(self):
#         return self.__in_features



class EfficientNetB0(BaseFeatureExtractor):
    def __init__(self, model_name='efficientnet', normalize=True, input_channels=3):
        super(EfficientNetB0, self).__init__()
        self.model_eff = EfficientNet.from_pretrained('efficientnet-b0')
        self.input_channels = input_channels
        if self.model_eff._conv_stem.in_channels != input_channels:
            Conv2d = self.model_eff._conv_stem.__class__
            old_conv = self.model_eff._conv_stem
            new_conv = Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(repeat_conv_weight(old_conv.weight, input_channels))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            self.model_eff._conv_stem = new_conv
        self.normalize = normalize
        self.mean = False
        self.std = False
        self.features = self.model_eff.extract_features
        self.__in_features = self.model_eff._fc.in_features

    def get_mean(self, device):
        if self.mean is False or self.mean.device != device:
            base_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            mean = np.tile(base_mean, int(math.ceil(self.input_channels / 3.0)))[:self.input_channels]
            self.mean = torch.from_numpy(mean.reshape((1, self.input_channels, 1, 1))).to(device)
        return self.mean

    def get_std(self, device):
        if self.std is False or self.std.device != device:
            base_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            std = np.tile(base_std, int(math.ceil(self.input_channels / 3.0)))[:self.input_channels]
            self.std = torch.from_numpy(std.reshape((1, self.input_channels, 1, 1))).to(device)
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean(x.device)) / self.get_std(x.device)
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def output_num(self):
        return self.__in_features

class DenseNet(BaseFeatureExtractor):
    def __init__(self, model_name='densenet', normalize=True, input_channels=3):
        super(DenseNet, self).__init__()
        self.model_dense = models.densenet121(pretrained=True)
        self.input_channels = input_channels
        self.model_dense.features.conv0 = adapt_conv2d_input_channels(
            self.model_dense.features.conv0,
            input_channels,
        )
        self.normalize = normalize
        self.mean = False
        self.std = False
        self.features = self.model_dense.features
        self.__in_features = self.model_dense.classifier.in_features

    def get_mean(self, device):
        if self.mean is False or self.mean.device != device:
            base_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            mean = np.tile(base_mean, int(math.ceil(self.input_channels / 3.0)))[:self.input_channels]
            self.mean = torch.from_numpy(mean.reshape((1, self.input_channels, 1, 1))).to(device)
        return self.mean

    def get_std(self, device):
        if self.std is False or self.std.device != device:
            base_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            std = np.tile(base_std, int(math.ceil(self.input_channels / 3.0)))[:self.input_channels]
            self.std = torch.from_numpy(std.reshape((1, self.input_channels, 1, 1))).to(device)
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean(x.device)) / self.get_std(x.device)
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def output_num(self):
        return self.__in_features


class Net_CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Net_CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x):
        x = self.fc(x)
        return x

class Net_CLS_C(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
        super(Net_CLS_C, self).__init__()
        #self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
        self.main = nn.Sequential(self.bottleneck,
                                  nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                                self.fc))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class Net_CLS_DC(nn.Module):
    def  __init__(self, in_dim, out_dim, bottle_neck_dim=None):
        super(Net_CLS_DC, self).__init__()
        if bottle_neck_dim is None:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
            )
        else:
            self.main = nn.Sequential(nn.Linear(in_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), \
                                      nn.Linear(bottle_neck_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), \
                                      nn.Linear(bottle_neck_dim, out_dim))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VGGBase(nn.Module):
    def __init__(self, input_channels=3):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        self.input_channels = input_channels
        if input_channels != 3:
            model_ft.features[0] = adapt_conv2d_input_channels(model_ft.features[0], input_channels)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        return x
    def output_num(self):
        return 100
