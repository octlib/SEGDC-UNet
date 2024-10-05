import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class up_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()
        self.ndf = 64
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = conv_block(ch_in=img_ch, ch_out=self.ndf)
        self.conv2 = conv_block(ch_in=self.ndf, ch_out=self.ndf * 2)
        self.conv3 = conv_block(ch_in=self.ndf * 2, ch_out=self.ndf * 2 * 2)
        self.conv4 = conv_block(ch_in=self.ndf * 2 * 2, ch_out=self.ndf * 2 * 2 * 2)
        self.conv5 = conv_block(ch_in=self.ndf * 2 * 2 * 2, ch_out=self.ndf * 2 * 2 * 2 * 2)

        self.up4 = up_block(ch_in=self.ndf * 2 * 2 * 2 * 2, ch_out=self.ndf * 2 * 2 * 2)
        self.up_conv4 = conv_block(ch_in=self.ndf * 2 * 2 * 2 * 2, ch_out=self.ndf * 2 * 2 * 2)

        self.up3 = up_block(ch_in=self.ndf * 2 * 2 * 2, ch_out=self.ndf * 2 * 2)
        self.up_conv3 = conv_block(ch_in=self.ndf * 2 * 2 * 2, ch_out=self.ndf * 2 * 2)

        self.up2 = up_block(ch_in=self.ndf * 2 * 2, ch_out=self.ndf * 2)
        self.up_conv2 = conv_block(ch_in=self.ndf * 2 * 2, ch_out=self.ndf * 2)

        self.up1 = up_block(ch_in=self.ndf * 2, ch_out=self.ndf)
        self.up_conv1 = conv_block(ch_in=self.ndf * 2, ch_out=self.ndf)

        self.conv1_1 = conv_block(ch_in=self.ndf, ch_out=output_ch)

    def forward(self, x):
        # x [none,3, 256, 256]
        x1 = self.conv1(x)  # [none,3,256,256]

        x1_ = self.Maxpool(x1)  # [none,64,128,128]
        x2 = self.conv2(x1_)  # [none,128,128,128]

        x2_ = self.Maxpool(x2)  # [none,128,64,64]
        x3 = self.conv3(x2_)  # [none,256,64,64]

        x3_ = self.Maxpool(x3)  # [none,256,32,32]
        x4 = self.conv4(x3_)  # [none,512,32,32]

        x4_ = self.Maxpool(x4)  # [none,512,16,16]
        x5 = self.conv5(x4_)  # [none,1024,16,16]

        u4_ = self.up4(x5)  # [none,1024,32,32]
        u4 = self.up_conv4(torch.cat([x4, u4_], dim=1))  # [none,512,32,32]

        u3_ = self.up3(u4)  # [none,512,64,64]
        u3 = self.up_conv3(torch.cat([x3, u3_], dim=1))  # [none,256,64,64]

        u2_ = self.up2(u3)  # [none,256,128,128]
        u2 = self.up_conv2(torch.cat([x2, u2_], dim=1))  # [none,128,128,128]

        u1_ = self.up1(u2)  # [none,128,256,256]
        u1 = self.up_conv1(torch.cat([x1, u1_], dim=1))  # [none,64,256,256]

        out = self.conv1_1(u1)  # [none,1,256,256]
        out = torch.sigmoid(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
model = U_Net()
print("Number of parameters in U_Net:", count_parameters(model))