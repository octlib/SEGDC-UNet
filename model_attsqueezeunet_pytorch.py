# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPool2D, BatchNormalization, ReLU, LeakyReLU, \
#     UpSampling2D, Activation, ZeroPadding2D, Lambda, AveragePooling2D, Reshape
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.activations import sigmoid
# from tensorflow.keras import Model, Sequential

import torch
from torch import nn


class FireModule(nn.Module):
    def __init__(self, fire_id, in_channels, squeeze, expand):
        super(FireModule, self).__init__()

        # self.fire = Sequential()
        # self.fire.add(Conv2D(squeeze, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal'))
        # self.fire.add(BatchNormalization(axis=-1))
        # self.left = Conv2D(expand, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')
        # self.right = Conv2D(expand, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels, squeeze, (1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(squeeze)
            # 与 tensorflow Conv2D (kernel_initializer='he_normal') 参数功能相当的凯明初始化？？？？？
        )
        self.left = nn.Sequential(
            nn.Conv2d(squeeze, expand, (1,1)),
            nn.ReLU()
            # 初始化 ？？？？？
        )
        self.right = nn.Sequential(
            nn.Conv2d(squeeze, expand, (3,3), padding=1),
            nn.ReLU()
            # 初始化 ？？？？
        )

    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        # x = tf.concat([left, right], axis=-1) # tensorflow中axis=-1为channel通道
        x = torch.cat([left, right], dim=1)     # torch中的dim=1为channel通道
        return x


class AttFireModule(nn.Module):
    def __init__(self,in_channels, filters, squeeze):
        super(AttFireModule, self).__init__()

        # self.fire = Sequential()
        # self.fire.add(Conv2D(squeeze, (1, 1), strides=(1, 1), use_bias=False, data_format='channels_last',
        #                      kernel_initializer='he_normal'))
        # self.fire.add(BatchNormalization(axis=-1))
        # self.left = Conv2D(filters, (1, 1), strides=(1, 1), use_bias=False, data_format='channels_last',
        #                    kernel_initializer='he_normal')
        # self.right = Conv2D(filters, (1, 1), strides=(1, 1), use_bias=False, data_format='channels_last',
        #                     kernel_initializer='he_normal')

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels, squeeze, (1,1), bias=False),
            nn.BatchNorm2d(squeeze)
            # 初始化 ？？？？
        )
        self.left = nn.Conv2d(squeeze, filters, (1,1), bias=False)
        # 初始化 ？？？

        self.right = nn.Conv2d(squeeze, filters, (1,1), bias=False)
        # 初始化 ？？？


    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        # x = tf.concat([left, right], axis=-1)
        x = torch.cat([left, right], dim=1)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_g, in_channels_x, filters):
        super(AttentionBlock, self).__init__()

        # self.w_g = Sequential()
        # self.w_g.add(Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False, data_format='channels_last',
        #                     kernel_initializer='he_normal'))
        # self.w_g.add(BatchNormalization())
        #
        # self.w_x = Sequential()
        # self.w_x.add(Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False, data_format='channels_last',
        #                     kernel_initializer='he_normal'))
        # self.w_x.add(BatchNormalization())
        #
        # self.psi = Sequential()
        # self.psi.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), use_bias=False, data_format='channels_last',
        #                     kernel_initializer='he_normal'))
        # self.psi.add(BatchNormalization())
        # self.psi.add(Activation("sigmoid"))
        #
        # self.relu = ReLU()

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels_g, filters, (1,1), bias=False),
            nn.BatchNorm2d(filters)
            # 初始化 ？？？
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels_x, filters, (1,1), bias=False),
            nn.BatchNorm2d(filters)
            # 初始化？？？
        )
        self.psi = nn.Sequential(
            nn.Conv2d(filters, 1, (1,1), bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            # 初始化 ????
        )
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UpsamplingBlock(nn.Module):
    def __init__(self,in_channels_x, in_channels_g, filters, fire_id, squeeze, expand, strides, deconv_ksize, att_filters):
        super(UpsamplingBlock, self).__init__()
        # self.upconv = Conv2DTranspose(filters, deconv_ksize, strides=strides, padding='same',
        #                               kernel_initializer='he_normal')
        # self.fire = FireModule(fire_id, squeeze, expand)
        # self.attention = AttentionBlock(att_filters)
        output_padding = 0 if strides[0]==1 else 1
        self.upconv = nn.ConvTranspose2d(in_channels_x, filters, deconv_ksize, stride=strides, padding=1, output_padding=output_padding)  # padding=1因为下面deconv_ksize=1
        # 初始化
        self.attention = AttentionBlock(filters, in_channels_g, att_filters)    # 输出通道=in_channels_g
        self.fire = FireModule(fire_id, in_channels_g+filters, squeeze, expand)


    def forward(self, x, g):
        d = self.upconv(x)
        x = self.attention(d, g)
        # d = tf.concat([x, d], axis=-1)
        d = torch.cat([x,d], dim=1)
        x = self.fire(d)
        return x


class AttSqueezeUNet(nn.Module):
    def __init__(self,in_channels=1, out_channels=1, dropout=False):  # filters equals to the number of classes
        super(AttSqueezeUNet, self).__init__()
        self.__dropout = dropout
        channel_axis = -1
        # self.conv_1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu',
        #                      kernel_initializer='he_normal')
        # self.max_pooling_1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # self.fire_1 = FireModule(2, 16, 64)
        # self.fire_2 = FireModule(3, 16, 64)
        # self.max_pooling_2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        #
        # self.fire_3 = FireModule(3, 32, 128)
        # self.fire_4 = FireModule(4, 32, 128)
        # self.max_pooling_3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        #
        # self.fire_5 = FireModule(5, 48, 192)
        # self.fire_6 = FireModule(6, 48, 192)
        # self.fire_7 = FireModule(7, 64, 256)
        # self.fire_8 = FireModule(8, 64, 256)

        # self.upsampling_1 = UpsamplingBlock(filters=192, fire_id=9, squeeze=48, expand=192, strides=(1, 1),
        #                                     deconv_ksize=3, att_filters=96)
        # self.upsampling_2 = UpsamplingBlock(filters=128, fire_id=10, squeeze=32, expand=128, strides=(1, 1),
        #                                     deconv_ksize=3, att_filters=64)
        # self.upsampling_3 = UpsamplingBlock(filters=64, fire_id=11, squeeze=16, expand=64, strides=(2, 2),
        #                                     deconv_ksize=3, att_filters=16)
        # self.upsampling_4 = UpsamplingBlock(filters=32, fire_id=12, squeeze=16, expand=32, strides=(2, 2),
        #                                     deconv_ksize=3, att_filters=4)
        # self.upsampling_5 = UpSampling2D(size=(2, 2))

        # self.conv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu',
        #                      kernel_initializer='he_normal')
        # self.upsampling_6 = UpSampling2D(size=(2, 2))
        # self.conv_3 = Conv2D(out_channels, (1, 1), activation='softmax' if out_channels > 1 else 'sigmoid')

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            # 初始化？？？？
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        self.fire_1 = FireModule(2, 64, 16, 64)
        self.fire_2 = FireModule(3, 64*2, 16, 64)
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        self.fire_3 = FireModule(3, 64*2, 32, 128)
        self.fire_4 = FireModule(4, 128*2, 32, 128)
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        self.fire_5 = FireModule(5, 128*2, 48, 192)
        self.fire_6 = FireModule(6, 192*2, 38, 192)
        self.fire_7 = FireModule(7, 192*2, 64, 256)
        self.fire_8 = FireModule(8, 256*2, 64, 256)

        self.upsampling_1 = UpsamplingBlock(in_channels_x=256*2, in_channels_g=192*2, filters=192, fire_id=9, squeeze=48, expand=192,
                                            strides=(1, 1), deconv_ksize=3, att_filters=96)
        self.upsampling_2 = UpsamplingBlock(in_channels_x=192*2, in_channels_g=128*2, filters=128, fire_id=10, squeeze=32, expand=128,
                                            strides=(1, 1), deconv_ksize=3, att_filters=64)
        self.upsampling_3 = UpsamplingBlock(in_channels_x=128*2, in_channels_g=64*2, filters=64, fire_id=11, squeeze=16, expand=64,
                                            strides=(2, 2), deconv_ksize=3, att_filters=16)
        self.upsampling_4 = UpsamplingBlock(in_channels_x=64*2, in_channels_g=64, filters=32, fire_id=12, squeeze=16, expand=32,
                                            strides=(2, 2), deconv_ksize=3, att_filters=4)
        self.upsampling_5 = nn.Upsample(scale_factor=(2, 2))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
            # 初始化???
        )
        self.upsampling_6 = nn.Upsample(scale_factor=(2, 2))
        if out_channels > 1:
            self.conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(1, 1)),
                nn.Softmax()
            )
        else:
            self.conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(1, 1)),
                nn.Sigmoid()
            )

    def forward(self, x):
        x0 = self.conv_1(x)
        x1 = self.max_pooling_1(x0)

        x2 = self.fire_1(x1)
        x2 = self.fire_2(x2)
        x2 = self.max_pooling_2(x2)

        x3 = self.fire_3(x2)
        x3 = self.fire_4(x3)
        x3 = self.max_pooling_3(x3)

        x4 = self.fire_5(x3)
        x4 = self.fire_6(x4)

        x5 = self.fire_7(x4)
        x5 = self.fire_8(x5)

        if self.__dropout:
            # x5 = Dropout(0.2)(x5)
            x5 = nn.Dropout(p=0.2)(x5)

        d5 = self.upsampling_1(x5, x4)
        d4 = self.upsampling_2(d5, x3)
        d3 = self.upsampling_3(d4, x2)
        d2 = self.upsampling_4(d3, x1)
        d1 = self.upsampling_5(d2)

        # d0 = tf.concat([d1, x0], axis=-1)
        d0 = torch.cat([d1, x0], dim=1)
        d0 = self.conv_2(d0)
        d0 = self.upsampling_6(d0)
        d = self.conv_3(d0)

        return d

if __name__ == '__main__':
        dcunet = AttSqueezeUNet()

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = AttSqueezeUNet()
print("Number of parameters in AttSqueezeUNet:", count_parameters(model))


