import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

class SCB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SCB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0,bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1,bias=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sobel_x1, self.sobel_y1 = get_sobel(in_channels, 1)

    def forward(self, x):
        y = run_sobel(self.sobel_x1, self.sobel_y1, x) # in_channels x x
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        return y


class Contour_Exectraction_Module(nn.Module):
    def __init__(self,out_feat = 2):
        super(Contour_Exectraction_Module, self).__init__()
        self.up1 = SCB(64,out_feat)
        self.up2 = SCB(128,out_feat)
        self.up3 = SCB(256,out_feat)
        self.conv1 = nn.Conv2d(out_feat * 3,out_feat,1,padding=0,bias=True)
    def forward(self, x1, x2, x3, feature=False):
        out1 = self.up1(x1)  # (bs,64,64,64) -> (bs 2 64 64)

        out2 = self.up2(x2)  # (bs,2,32,32)
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)  # (1,1,32,32)

        out3 = self.up3(x3)  # (1 2 16 16)
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear', align_corners=True)  # (1,1,32,32)

        edge = torch.cat((out1,out2,out3),dim=1)
        edge = self.conv1(edge)
        return edge


class Contour_Exectraction_Module2(nn.Module):  ######## contour preservation module

    def __init__(self, abn=nn.BatchNorm2d, in_fea=[64, 128, 256], mid_fea=64, out_fea=2):
        super(Contour_Exectraction_Module2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge = self.conv5(edge)

        return edge


if __name__ == '__main__':
    net = Contour_Exectraction_Module2().cuda()
    input1 = torch.randn(1, 64, 64, 64).cuda()
    input2 = torch.randn(1, 128, 32, 32).cuda()
    input3 = torch.randn(1, 256, 16, 16).cuda()
    output = net(input1, input2, input3)
    print(output)
