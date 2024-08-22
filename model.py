import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import cv2
import numpy as np


class DRS(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding):
        super(DRS, self).__init__()
        self.asb = ASB(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=padding)
        self.anb = ANB(in_channels=output_size)

    def forward(self, x):
        std_feat = self.asb(x)
        norm_feat = self.anb(std_feat)

        return norm_feat


class MASEblock(nn.Module): #SE with MAX
    def __init__(self, in_channels, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class MISEblock(nn.Module): #SE with MIN
    def __init__(self, in_channels, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = -self.squeeze(-x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class ANB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.maseblock = MASEblock(in_channels)
        self.miseblock = MISEblock(in_channels)

    def forward(self, x):

        im_h = self.maseblock(x)
        im_l = self.miseblock(x)

        me = torch.tensor(0.00001, dtype=torch.float32).cuda()

        x = (x - im_l) / torch.maximum(im_h - im_l, me)
        x = torch.clip(x, 0.0, 1.0)

        return x


class ASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class FSA(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(FSA, self).__init__()
        inNet_dim = input_dim + 3
        #input part
        self.feat1 = ConvBlock(inNet_dim, dim, 3, 1, 1)
        self.feat2 = ConvBlock(dim, 2 * dim, 3, 1, 1)
        #FS
        self.locin = ConvBlock(dim, dim, 1, 1, 0)
        self.loca = DRS(dim, dim, 3, 1)
        self.lres = ResBlock(dim, dim, 1, 1, 0)
        self.lseres = SERESLayer(channel=2 * dim)
        self.loc = ConvBlock(2 * dim, dim, 1, 1, 0)
        #DR
        self.globin = ConvBlock(2 * dim, 2 * dim, 3, 1, 1)
        self.gres = MultiBlock(2 * dim, 2 * dim)
        self.gseres = SERESLayer(channel=4 * dim)
        self.glob = ConvBlock(4 * dim, 2 * dim, 1, 1, 0)
        self.gdim = ConvBlock(2 * dim, dim, 3, 1, 1)
        #FS+DR
        self.lg = MSERESLayer(channel=2 * dim)
        self.Light_map = ConvBlock(2 * dim, dim, 3, 1, 1)
        #DRS
        self.color_in = DRS(3, 3, 3, 1)
        self.Color_map = ConvBlock(3, dim, 3, 1, 1)
        #output
        self.feature = ConvBlock(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_ori, tar=None):
        #feature extraction
        xh = (255 * x_ori).type('torch.cuda.ByteTensor')
        xhist = torchvision.transforms.functional.equalize(xh)
        xhist = (xhist / 255).type('torch.cuda.FloatTensor')
        x_ori = x_ori.type('torch.cuda.HalfTensor')
        xhist = xhist.type('torch.cuda.HalfTensor')
        x_in = torch.cat((x_ori, xhist), 1)
        ##################################################
        Local_feat = self.feat1(x_in)
        Global_feat = self.feat2(Local_feat)
        ##################################################
        loc_in = self.locin(Local_feat)
        locaa = self.loca(Local_feat)
        l_res = self.lres(locaa)
        lse_in = torch.cat([loc_in, l_res], dim=1)
        lse_out = self.lseres(lse_in)
        lse_i = self.loc(lse_out)
        loc_out = loc_in + lse_i
        ##################################################
        glob_in = self.globin(Global_feat)
        g_res = self.gres(Global_feat)
        gse_in = torch.cat([glob_in, g_res], dim=1)
        gse_out = self.gseres(gse_in)
        glob_se_out = self.glob(gse_out)
        glob_out = glob_in + glob_se_out
        glob_totaL = self.gdim(glob_out)
        ##################################################
        Brighten_map_in = torch.cat([loc_out, glob_totaL], dim=1)
        briin = self.lg(Brighten_map_in)
        briout = Brighten_map_in + briin
        Brighten_map_out = self.Light_map(briout)
        ##################################################
        Color_map_in = self.color_in(x_ori)
        Color_map_out = self.Color_map(Color_map_in)
        ##################################################
        BC = Brighten_map_out * Color_map_out
        feature_out = self.feature(BC)
        pred = self.out(feature_out)

        return pred


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class SERESLayer(nn.Module): #SE with Avgpooling
    def __init__(self, channel, reduction=16):
        super(SERESLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #avgpooling
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
        y = x * y.expand_as(x)
        return y


class MSERESLayer(nn.Module): #SE with Maxpooling
    def __init__(self, channel, reduction=16):
        super(MSERESLayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1) #maxpooling
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
        y = x * y.expand_as(x)
        return y


class ResBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return out


class MultiBlock(torch.nn.Module): #Multi attention
    def __init__(self, input_size, output_size, bias=True):
        super(MultiBlock, self).__init__()
        self.lev1 = nn.Sequential(
            nn.Conv2d(input_size, input_size, 3, 1, 1),
            nn.ReLU()
        )
        self.lev2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, 3, 2, 1), #downscale x2
            nn.Conv2d(input_size, input_size, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size, input_size, 2, 2) #upscale x2
        )
        self.lev3 = nn.Sequential(
            nn.Conv2d(input_size, input_size, 3, 4, 1), #downscale x4
            nn.Conv2d(input_size, input_size, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size, input_size, 4, 4) #upscale x4
        )
        self.oconv = nn.Sequential(
            nn.Conv2d(4 * input_size, output_size, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        l1 = self.lev1(x) #x1
        l2 = self.lev2(x) #x2
        l3 = self.lev3(x) #x4
        total_lev = torch.cat([x, l1, l2, l3], dim=1)
        out = self.oconv(total_lev)
        return out