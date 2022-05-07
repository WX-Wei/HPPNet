from lib2to3.pgen2.token import NT_OFFSET
from math import sin
from multiprocessing import pool
import torch

import torch.nn as nn
import torch.nn.functional as F
import nnAudio
import torchaudio

import torchvision
import matplotlib.pyplot as plt
import os

from .constants import *
from .layers import WaveformToHarmgram
from .lstm import BiLSTM

# multiple rate dilated causal convolution
class MRDC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_list = [0, 12, 19, 24, 28, 31, 34, 36]):
        super().__init__()
        self.dilation_list = dilation_list
        self.conv_list = []
        for i in range(len(dilation_list)):
            self.conv_list += [nn.Conv2d(in_channels, out_channels, kernel_size = [1, 1])]
        self.conv_list = nn.ModuleList(self.conv_list)
        
    def forward(self, specgram):
        # input [b x C x T x n_freq]
        # output: [b x C x T x n_freq] 
        specgram
        dilation = self.dilation_list[0]
        y = self.conv_list[0](specgram)
        y = F.pad(y, pad=[0, dilation])
        y = y[:, :, :, dilation:]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](specgram)
            # => [b x T x (n_freq + dilation)]
            # x = F.pad(x, pad=[0, dilation])
            x = x[:, :, :, dilation:]
            n_freq = x.size()[3]
            y[:, :, :, :n_freq] += x

        return y

class FrqeBinLSTM(nn.Module):
    def __init__(self, channel_in, channel_out, lstm_size) -> None:
        super().__init__()

        self.channel_out = channel_out

        self.lstm = BiLSTM(channel_in, lstm_size//2)
        self.linear = nn.Linear(lstm_size, channel_out)

    def forward(self, x):
        # inputs: [b x c_in x T x freq]
        # outputs: [b x c_out x T x freq]

        b, c_in, t, n_freq = x.size() 

        # => [b x freq x T x c_in] 
        x = torch.permute(x, [0, 3, 2, 1])

        # => [(b*freq) x T x c_in]
        x = x.reshape([b*n_freq, t, c_in])
        # => [(b*freq) x T x lstm_size]
        x = self.lstm(x)
        # => [(b*freq) x T x c_out]
        x = self.linear(x)
        # => [b x freq x T x c_out]
        x = x.reshape([b, n_freq, t, self.channel_out])
        # => [b x c_out x T x freq]
        x = torch.permute(x, [0, 3, 2, 1])
        x = torch.sigmoid(x)
        return x

class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 48])
        self.conv_2 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 76])
        self.conv_3 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 96])
        self.conv_4 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 111])
        self.conv_5 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 124])
        self.conv_6 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 135])
        self.conv_7 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 144])
        self.conv_8 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 152])
    def forward(self, x):
        x = self.conv_1(x) + self.conv_2(x) + self.conv_3(x) + self.conv_4(x) +\
            self.conv_5(x) + self.conv_6(x) + self.conv_7(x) + self.conv_8(x)
        x = torch.relu(x)
        return x


class CNNTrunk(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                nn.BatchNorm2d(channel_out),
                # nn.InstanceNorm2d(channel_out)
            )

    def __init__(self, c_in = 2, c_har = 16,  embedding = 128, trunk_type='HD-Conv', fixed_dilation = 24) -> None:
        super().__init__()

        self.block_1 = self.get_conv2d_block(c_in, c_har, kernel_size=7)
        self.block_2 = self.get_conv2d_block(c_har, c_har, kernel_size=7)
        self.block_2_5 = self.get_conv2d_block(c_har, c_har, kernel_size=7)

        c3_out = embedding

        # self.block_3 = MRDC_Conv(c_har, 64, dilation_list=[48, 76, 96, 111, 124, 135, 144, 152, 159, 166])
        
        if(trunk_type == 'Conv'):
            self.conv_3 = nn.Conv2d(c_har, c3_out, [1, 3], padding='same')
        elif(trunk_type == 'HD-Conv'):
            self.conv_3 = HarmonicDilatedConv(c_har, c3_out)
        elif(trunk_type == 'SD-Conv'):
            self.conv_3 = nn.Conv2d(c_har, c3_out, [1, 8], padding='same', dilation=[1, fixed_dilation])

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48])
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        # self.conv_9 = nn.Conv2d(c3_out, 64,1)
        # self.conv_10 = nn.Conv2d(64, 1, 1)

    def forward(self, log_gram_db):
        # inputs: [b x 2 x T x n_freq]
        # outputs: [b x T x 88]


        # img_path = 'logspecgram_preview.png'
        # if not os.path.exists(img_path):
        #     img = torch.permute(log_gram_db, [2, 0, 1]).reshape([352, 640*4]).detach().cpu().numpy()
        #     # x_grid = torchvision.utils.make_grid(x.swapaxes(0, 1), pad_value=1.0).swapaxes(0, 2).detach().cpu().numpy()
        #     # plt.imsave(img_path, (x_grid+80)/100)
        #     plt.imsave(img_path, img)

        # => [b x 1 x T x 352]
        # x = torch.unsqueeze(log_gram_db, dim=1)

        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) + x
        x = self.block_7(x) + x
        x = self.block_8(x) + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)

        return x