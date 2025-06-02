import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model.MobileNetV2 import mobilenet_v2
from torch.nn import Parameter

#-------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
#-------------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
#-------------------------------------------------------------------------------
class NSB(nn.Module):
    def __init__(self, in_channel=320):
        super(NSB, self).__init__()

        self.channel_attention = ChannelAttention(in_channel)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)
        self.conv_2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)  # Padding 1
        self.dilated_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=3, dilation=3)  # Dilation 3, Padding 3

    def forward(self, x):
        x = self.channel_attention(x) * x
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        enhaned_edgess = weight * x + x
        enhaned_edgess = self.PReLU(enhaned_edgess)
        normal_output = self.conv_2(x)
        dilated_output = self.dilated_conv(x)
        output = normal_output + dilated_output
        final = torch.add(output, enhaned_edgess)
        return final
#-----------------------------------------------------------------------
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
#----------------------------------------------------------------------
def decoder(inp, oup):
    return nn.Sequential(
        conv_dw(inp, oup, 1),
        nn.ConvTranspose2d(oup, oup, kernel_size=4,
                           stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
#-----------------------------------------------------------------------
class AFB(nn.Module):
    def __init__(self, mid_ch,out_ch):
        super(AFB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid())

    def forward(self, input_high, input_low):
        mid_high=self.global_pooling(input_high)
        weight_high=self.conv1(mid_high)
        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)
        return input_high.mul(weight_high)+input_low.mul(weight_low)
#-------------------------------------------------------------------
class nucleinet(nn.Module):
    def __init__(self,pretrained=True):
        super(nucleinet, self).__init__()
        # Backbone model
        self.backbone = mobilenet_v2(pretrained)
        self.nsb5 = NSB(320)
        self.nsb4 = NSB(96)
        self.nsb3 = NSB(32)
        self.nsb2 = NSB(24)
        self.nsb1 = NSB(16)

        self.dec5 = decoder(320, 96)
        self.dec4 = decoder(96, 32)
        self.dec3 = decoder(32, 24)
        self.dec2 = decoder(24, 16)
        self.dec1 = decoder(16, 8)

        self.afb5 = AFB(320, 320)
        self.afb4 = AFB(96, 96)
        self.afb3 = AFB(32, 32)
        self.afb2 = AFB(24, 24)
        self.afb1 = AFB(16, 16)

        self.outc = nn.Conv2d(8, 1, kernel_size=1, padding=0)
        self.last_activation = nn.Sigmoid()

    def forward(self, input):
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        c5 = self.nsb5(conv5)
        afb5 = self.afb5(conv5, c5)
        output_dec5 = self.dec5(afb5)
        c4 = self.nsb4(conv4)
        afb4 = self.afb4(output_dec5, c4)
        output_dec4 = self.dec4(afb4)
        c3 = self.nsb3(conv3)
        afb3 = self.afb3(output_dec4, c3)
        output_dec3 = self.dec3(afb3)
        c2 = self.nsb2(conv2)
        afb2 = self.afb2(output_dec3, c2)
        output_dec2 = self.dec2(afb2)
        c1 = self.nsb1(conv1)
        afb1 = self.afb1(output_dec2, c1)
        output_dec1 = self.dec1(afb1)
        final_output = self.outc(output_dec1)
        return final_output


if __name__ == '__main__':
    num_classes = 1
    in_batch, inchannel, in_h, in_w = 10, 3, 512, 512
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = nucleinet()
    out = net(x)
    print(out.shape)
