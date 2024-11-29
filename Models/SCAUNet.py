import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EncoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderLayer, self).__init__()

        self.layer_first = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            SELayer(in_ch),  # 添加注意力模块
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.layer_next = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            SELayer(in_ch),  # 添加注意力模块
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer_first(x)
        out = self.layer_next(out)
        # out = self.relu(out)

        return out




class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x



class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # inplace为True,将会改变输入的数据;否则不改变原输入,只产生新的输出

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class XYA(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(XYA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(inp, inp // groups)

        self.conv0 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.conv4 = nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = self.conv0(x).sigmoid()
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h
        y = self.conv4(y).sigmoid()

        return y

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Reshape tensors for matrix multiplication
        batch_size, channels, height, width = query.size()
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size, channels, -1).transpose(1, 2)
        value = value.view(batch_size, channels, -1)

        # Compute attention weights
        attention_weights = torch.bmm(query, key)
        attention_weights = attention_weights / (channels ** 0.5)
        attention_weights = self.softmax(attention_weights)

        # Apply attention weights to the value
        attended_value = torch.bmm(attention_weights, value)
        attended_value = attended_value.view(batch_size, channels, height, width)

        # Add attended_value to the input (residual connection)
        output = x + attended_value
        # output = attended_value

        return output


def CBR(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class HDUC(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(HDUC, self).__init__()
        self.conv1 = CBR(in_ch, out_ch, 1)
        self.dconv1 = CBR(out_ch, out_ch // 2, 2)
        self.dconv2 = CBR(out_ch // 2, out_ch // 2, 4)
        self.dconv3 = CBR(out_ch, out_ch, 2)
        self.conv2 = CBR(out_ch * 2, out_ch, 1)
        self.attention = Scaled_Dot_Product_Attention(out_ch, out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.attention(x1)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))
        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src


class SCAUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dim=64, deep_supervision=True, **kwargs):
        super(SCAUNet, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [dim, dim*2, dim*4, dim*8, dim*16]

        # 编码层
        self.Encode_layer1 = EncoderLayer(in_ch, filters[0])
        self.Encode_layer2 = EncoderLayer(filters[0], filters[1])
        self.Encode_layer3 = EncoderLayer(filters[1], filters[2])
        self.Encode_layer4 = EncoderLayer(filters[2], filters[3])

        # 下采样的最大池化层
        self.encode_down = nn.MaxPool2d(2, 2)

        # 左边的特征增强模块
        self.xya1 = XYA(filters[0], filters[0])
        self.xya2 = XYA(filters[1], filters[1])
        self.xya3 = XYA(filters[2], filters[2])
        self.xya4 = XYA(filters[3], filters[3])

        # 第一个HDUC模块
        self.dc_conv = HDUC(filters[3], filters[4])
        # self.dc_conv = conv_block(filters[3], filters[4])

        # 右边上采样部分以及解码(第一个解码是HDUC模块)
        self.up5 = Up(filters[4], filters[3])
        self.up_conv5 = HDUC(filters[4], filters[3])
        # self.up_conv5 = conv_block(filters[4], filters[3])

        self.up4 = Up(filters[3], filters[2])
        self.up_conv4 = conv_block(filters[3], filters[2])

        self.up3 = Up(filters[2], filters[1])
        self.up_conv3 = conv_block(filters[2], filters[1])

        self.up2 = Up(filters[1], filters[0])
        self.up_conv2 = conv_block(filters[1], filters[0])

        # 卷积头，解码出最后结果
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # 深监督部分的解码-----------------------------------------------------------------------------------
        self.conv5 = nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)
        # ------------------------------------------------------------------------------------------------



    def forward(self, x):

        e1 = self.Encode_layer1(x)

        e2 = self.encode_down(e1)
        e2 = self.Encode_layer2(e2)

        e3 = self.encode_down(e2)
        e3 = self.Encode_layer3(e3)

        e4 = self.encode_down(e3)
        e4 = self.Encode_layer4(e4)

        e1_s = self.xya1(e1)
        e2_s = self.xya2(e2)
        e3_s = self.xya3(e3)
        e4_s = self.xya4(e4)

        e4_d = self.encode_down(e4_s)
        # e4_d = self.encode_down(e4)
        dc5 = self.dc_conv(e4_d)

        d5 = self.up5(dc5)
        # d5 = torch.cat((e4_s, d5), dim=1)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        # d4 = torch.cat((e3_s, d4), dim=1)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        # d3 = torch.cat((e2_s, d3), dim=1)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1_s, d2), dim=1)
        # d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        out = self.Conv(d2)

        d_s1 = self.conv1(d2)
        d_s2 = self.conv2(d3)
        d_s2 = _upsample_like(d_s2, d_s1)
        d_s3 = self.conv3(d4)
        d_s3 = _upsample_like(d_s3, d_s1)
        d_s4 = self.conv4(d5)
        d_s4 = _upsample_like(d_s4, d_s1)
        d_s5 = self.conv5(dc5)
        d_s5 = _upsample_like(d_s5, d_s1)
        if self.deep_supervision:
            outs = [d_s1, d_s2, d_s3, d_s4, d_s5, out]
        else:
            outs = out

        return outs


if __name__=='__main__':
    x = torch.randn(4, 3, 512, 512)
    model = SCAUNet()
    y = model(x)
    print(model(x))




