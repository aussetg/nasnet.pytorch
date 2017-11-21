import torch
import torch.nn as nn
from torch.nn.functional import relu
from .droppath import DropPath

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class TwoSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(TwoSeparables, self).__init__()
        self.separable_0 = SeparableConv2d(in_channels, in_channels, dw_kernel, dw_stride, dw_padding, bias=bias)
        self.bn_0 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.separable_1 = SeparableConv2d(in_channels, out_channels, dw_kernel, 1, dw_padding, bias=bias)
        self.bn_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = relu(x)
        x = self.separable_0(x)
        x = self.bn_0(x)
        x = relu(x)
        x = self.separable_1(x)
        x = self.bn_1(x)
        return x

class ResizeCell0(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels):
        super(ResizeCell0, self).__init__()
        self.pool_left_0 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_left_0 = nn.Conv2d(in_channels_h, out_channels//2, 1, stride=1, bias=False)

        self.pool_left_1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_left_1 = nn.Conv2d(in_channels_h, out_channels//2, 1, stride=1, bias=False)

        self.bn_left = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

        self.conv_right = nn.Conv2d(in_channels_x, out_channels, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x, h):
        h = relu(h)

        h_0 = self.pool_left_0(h)
        h_0 = self.conv_left_0(h_0)

        h_1 = self.pool_left_1(h)
        h_1 = self.conv_left_1(h_1)

        h = torch.cat([h_0, h_1], 1)
        h = self.bn_left(h)

        x = relu(x)
        x = self.conv_right(x)
        x = self.bn_right(x)

        return x, h

class ResizeCell1(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels):
        super(ResizeCell1, self).__init__()
        self.conv_left = nn.Conv2d(in_channels_x, out_channels, 1, stride=1, bias=False)
        self.bn_left = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

        self.conv_right = nn.Conv2d(in_channels_h, out_channels, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x, h):
        x = relu(x)
        x = self.conv_left(x)
        x = self.bn_left(x)

        h = relu(h)
        h = self.conv_right(h)
        h = self.bn_right(h)

        return x, h


class ReductionCell(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels, resize_cell=ResizeCell1, keep_prob=0.9):
        super(ReductionCell, self).__init__()

        self.resize = resize_cell(in_channels_x, in_channels_h, out_channels)

        self.comb_iter_0_left = DropPath(TwoSeparables(out_channels, out_channels, 7, 2, 3, bias=False), keep_prob)
        self.comb_iter_0_right = DropPath(TwoSeparables(out_channels, out_channels, 5, 2, 2, bias=False), keep_prob)

        self.comb_iter_1_left = DropPath(nn.MaxPool2d(3, stride=2, padding=1), keep_prob)
        self.comb_iter_1_right = DropPath(TwoSeparables(out_channels, out_channels, 7, 2, 3, bias=False), keep_prob)

        self.comb_iter_2_left = DropPath(nn.AvgPool2d(3, stride=2, padding=1), keep_prob)
        self.comb_iter_2_right = DropPath(TwoSeparables(out_channels, out_channels, 5, 2, 2, bias=False), keep_prob)

        self.comb_iter_3_left = DropPath(nn.MaxPool2d(3, stride=2, padding=1), keep_prob)
        self.comb_iter_3_right = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)

        self.comb_iter_4_left = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)

    def forward(self, x, h):
        prev = x

        x, h = self.resize(x, h)

        comb_iter_0_left = self.comb_iter_0_left(h)
        comb_iter_0_right = self.comb_iter_0_right(x)
        comb_iter_0 = comb_iter_0_left + comb_iter_0_right

        comb_iter_1_left = self.comb_iter_1_left(x)
        comb_iter_1_right = self.comb_iter_1_right(h)
        comb_iter_1 = comb_iter_1_left + comb_iter_1_right

        comb_iter_2_left = self.comb_iter_2_left(x)
        comb_iter_2_right = self.comb_iter_2_right(h)
        x_comb_iter_2 = comb_iter_2_left + comb_iter_2_right

        comb_iter_3_left = self.comb_iter_3_left(x)
        comb_iter_3_right = self.comb_iter_3_right(comb_iter_0)
        comb_iter_3 = comb_iter_3_left + comb_iter_3_right

        comb_iter_4_left = self.comb_iter_4_left(comb_iter_0)
        comb_iter_4 = comb_iter_4_left + comb_iter_1

        return torch.cat([comb_iter_1, x_comb_iter_2, comb_iter_3, comb_iter_4], 1), prev

class NormalCell(nn.Module):
    def __init__(self, in_channels_x, in_channels_h, out_channels, resize_cell=ResizeCell1, keep_prob=0.9):
        super(NormalCell, self).__init__()
        self.adjust = resize_cell(in_channels_x, in_channels_h, out_channels)

        self.comb_iter_0_left = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)

        self.comb_iter_1_left = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)
        self.comb_iter_1_right = DropPath(TwoSeparables(out_channels, out_channels, 5, 1, 2, bias=False), keep_prob)

        self.comb_iter_2_left = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)

        self.comb_iter_3_left = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)
        self.comb_iter_3_h = DropPath(nn.AvgPool2d(3, stride=1, padding=1), keep_prob)

        self.comb_iter_4_left = DropPath(TwoSeparables(out_channels, out_channels, 5, 1, 2, bias=False), keep_prob)
        self.comb_iter_4_right = DropPath(TwoSeparables(out_channels, out_channels, 3, 1, 1, bias=False), keep_prob)

    def forward(self, x, h):
        prev = x

        x, h = self.adjust(x, h)

        comb_iter_0_left = self.comb_iter_0_left(x)
        comb_iter_0 = comb_iter_0_left + x

        comb_iter_1_left = self.comb_iter_1_left(h)
        comb_iter_1_right = self.comb_iter_1_right(x)
        comb_iter_1 = comb_iter_1_left + comb_iter_1_right

        comb_iter_2_left = self.comb_iter_2_left(x)
        comb_iter_2 = comb_iter_2_left + h

        comb_iter_3_left = self.comb_iter_3_left(h)
        comb_iter_3_right = self.comb_iter_3_h(h)
        comb_iter_3 = comb_iter_3_left + comb_iter_3_right

        comb_iter_4_left = self.comb_iter_4_left(h)
        comb_iter_4_right = self.comb_iter_4_right(h)
        comb_iter_4 = comb_iter_4_left + comb_iter_4_right

        return torch.cat([x, comb_iter_0, comb_iter_1, comb_iter_2, comb_iter_3, comb_iter_4], 1), prev
