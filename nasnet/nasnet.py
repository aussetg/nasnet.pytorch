import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.functional import relu
from .layers import NormalCell, ReductionCell, ResizeCell0, ResizeCell1

class NASNet(nn.Module):
    def __init__(self, stem_filters, normals, filters, scaling, num_classes, use_aux=True, pretrained=True):
        super(NASNet, self).__init__()
        self.normals = normals
        self.use_aux = use_aux
        self.num_classes = num_classes

        self.stemcell = nn.Sequential(
            nn.Conv2d(3, stem_filters, kernel_size=3, stride=2),
            nn.BatchNorm2d(stem_filters, eps=0.001, momentum=0.1, affine=True)
        )

        self.reduction1 = ReductionCell(in_channels_x=stem_filters,
                                        in_channels_h=stem_filters,
                                        out_channels=int(filters * scaling ** (-2)),
                                        resize_cell=ResizeCell1)
        self.reduction2 = ReductionCell(in_channels_x=int(4*filters * scaling ** (-2)),
                                        in_channels_h=stem_filters,
                                        out_channels=int(filters * scaling ** (-1)),
                                        resize_cell=ResizeCell0)

        x_channels = int(4*filters * scaling ** (-1))
        h_channels = int(4*filters * scaling ** (-2))

        self.add_module('normal_block1_0',
                        NormalCell(in_channels_x=x_channels,
                                   in_channels_h=h_channels,
                                   out_channels=filters,
                                   resize_cell=ResizeCell0,
                                   keep_prob=0.9))
        # TODO: Can we do that in a cleaner way?
        h_channels = x_channels
        x_channels = 6*filters
        for i in range(normals-1):
            self.add_module('normal_block1_{}'.format(i+1),
                            NormalCell(in_channels_x=x_channels,
                                       in_channels_h=h_channels,
                                       out_channels=filters,
                                       resize_cell=ResizeCell1,
                                       keep_prob=0.9))
            h_channels = x_channels
            x_channels = 6*filters

        self.reduction3 = ReductionCell(in_channels_x=x_channels,
                                        in_channels_h=h_channels,
                                        out_channels=filters * scaling)

        h_channels = x_channels
        x_channels = 4 * filters * scaling

        self.add_module('normal_block2_0',
                        NormalCell(in_channels_x=x_channels,
                                   in_channels_h=h_channels,
                                   out_channels=filters*scaling,
                                   resize_cell=ResizeCell0,
                                   keep_prob=0.9))
        h_channels = x_channels
        x_channels = 6 * filters * scaling
        for i in range(normals - 1):
            self.add_module('normal_block2_{}'.format(i + 1),
                            NormalCell(in_channels_x=x_channels,
                                       in_channels_h=h_channels,
                                       out_channels=filters*scaling,
                                       resize_cell=ResizeCell1, keep_prob=0.9))
            h_channels = x_channels
            x_channels = 6 * filters * scaling

        self.reduction4 = ReductionCell(in_channels_x=x_channels,
                                        in_channels_h=h_channels,
                                        out_channels=filters * scaling ** 2)

        h_channels = x_channels
        x_channels = 4 * filters * scaling ** 2

        self.add_module('normal_block3_0',
                        NormalCell(in_channels_x=x_channels,
                                   in_channels_h=h_channels,
                                   out_channels=filters * scaling ** 2,
                                   resize_cell=ResizeCell0, keep_prob=0.9))
        h_channels = x_channels
        x_channels = 6 * filters * scaling ** 2
        for i in range(normals - 1):
            self.add_module('normal_block3_{}'.format(i + 1),
                            NormalCell(in_channels_x=x_channels,
                                       in_channels_h=h_channels,
                                       out_channels=filters * scaling ** 2,
                                       resize_cell=ResizeCell1,
                                       keep_prob=0.9))
            h_channels = x_channels
            x_channels = 6 * filters * scaling ** 2

        self.avg_pool_0 = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout_0 = nn.Dropout()
        self.fc = nn.Linear(x_channels, self.num_classes)

    def features(self, x):
        x = self.stemcell(x)

        x, h = self.reduction1(x, x)
        x, h = self.reduction2(x, h)

        for i in range(self.normals):
            x, h = self._modules['normal_block1_{}'.format(i)](x, h)

        x, h = self.reduction3(x, h)

        for i in range(self.normals):
            x, h = self._modules['normal_block2_{}'.format(i)](x, h)

        # Should we check for training or not ?
        if self.use_aux and self.training:
            x_aux = x

        x, h = self.reduction4(x, h)

        for i in range(self.normals):
            x, h = self._modules['normal_block3_{}'.format(i)](x, h)

        if self.use_aux and self.training:
            return x, x_aux
        else:
            return x

    def classifier(self, x):
        x = relu(x)
        x = self.avg_pool_0(x)
        x = x.view(-1, self.fc.in_features)
        x = self.dropout_0(x)
        x = self.fc(x)
        return x

    def aux_classifier(self, x):
        x = relu(x)
        x = self.avg_pool_0(x)
        x = x.view(-1, self.fc.in_features)
        x = self.dropout_0(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        if self.use_aux:
            x, x_b = self.features(x)
            x = self.classifier(x)
            x_b = self.aux_classifier(x_b)
            return x, x_b
        else:
            x = self.features(x)
            x = self.classifier(x)
            return x


def nasnetmobile(num_classes=1000, pretrained=False):
    return NASNet(32, 4, 44, 2, num_classes=num_classes, use_aux=True, pretrained=pretrained)

def nasnetlarge(num_classes=1000, pretrained=False):
    return NASNet(96, 6, 168, 2, num_classes=num_classes, use_aux=True, pretrained=pretrained)