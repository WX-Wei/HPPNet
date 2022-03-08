from math import sin
from multiprocessing import pool
import torch

import torch.nn as nn
import torch.nn.functional as F

from .constants import *

from .layers import WaveformToHarmgram

class HarmSpecgramConvBlock(nn.Module):
    def get_conv3d_block(self, channel_in,channel_out, pool_size = [1, 2, 2]):
        return nn.Sequential( 
            nn.Conv3d(channel_in, channel_out, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool3d(pool_size),
            nn.BatchNorm3d(channel_out),
            )
    def dil_conv2d_block(self, dil_rate):
        return nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=[1, 3], padding=[0, dil_rate], dilation=dil_rate),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding='same', padding_mode='replicate'),
            # nn.ReLU(),
        )
    def __init__(self, output_bins) -> None:
        super().__init__()
        self.device = DEFAULT_DEVICE

        sr = SAMPLE_RATE
        n_freq = WINDOW_LENGTH // 2
        n_har = N_HAR
        bins_per_semitone = BINS_PER_SEMITONE
        hop_length = HOP_LENGTH
        self.WaveformToHarmgram = WaveformToHarmgram(sr, n_freq, n_har, bins_per_semitone, hop_length=hop_length, device=self.device)

        self.conv3d_block_1 = self.get_conv3d_block(1, 16)
        self.conv3d_block_2 = self.get_conv3d_block(16, 32)
        self.conv3d_block_3 = self.get_conv3d_block(32, 64, pool_size=[1, 1, 2])
        # self.conv3d_block_4 = self.get_conv3d_block(64, 64, pool_size=[1, 1, 2])
        # self.conv3d_block_5 = self.get_conv3d_block(64, 64, pool_size=[1, 1, 2])
        # self.conv3d_block_6 = self.get_conv3d_block(64, 64, pool_size=[1, 1, 2])

        self.dil_block = nn.Sequential(
            self.dil_conv2d_block(12),
            self.dil_conv2d_block(19),
            self.dil_conv2d_block(24),
            self.dil_conv2d_block(28),
            # self.dil_conv2d_block(31),
            self.dil_conv2d_block(36),
        )

        # self.dil_conv2d_1 = 
        # self.dil_conv2d_2 = 
        # self.dil_conv2d_3 = 
        # self.dil_conv2d_4 = 


        self.linear_1 = nn.Linear(64, 32)
        self.linear_2 = nn.Linear(32, 1)

        # self.linear3 = nn.Linear(88, output_bins) # 16 * MODEL_COMPLEXITY


    def forward(self, waveforms):
        # inputs: [b x (hop_length*T)]
        # outputs: [b x T x 88]

        waveforms = waveforms.to(self.device)

        # => [b x T x (88*bins_per_semitone) x n_har]
        har_gram = self.WaveformToHarmgram(waveforms).float()

        # => [b x 1 x T x (88*bins_per_semitone) x n_har]
        x = torch.unsqueeze(har_gram, dim=1)

        x = self.conv3d_block_1(x)
        x = self.conv3d_block_2(x)
        x = self.conv3d_block_3(x)
        # => # [b x ch x T x 88 x 1]
        # x = self.conv3d_block_4(x)
        # x = self.conv3d_block_5(x)
        # x = self.conv3d_block_6(x)
        # => [b x ch x T x 88]
        x = torch.squeeze(x, dim = 4)

        x = self.dil_block(x)
        
        # x = self.dil_conv2d_1(x)
        # x = torch.relu(x)
        # x = self.dil_conv2d_2(x)
        # x = torch.relu(x)
        # x = self.dil_conv2d_3(x)
        # x = torch.relu(x)
        # => [b x T x 88 x ch]
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.linear_1(x)
        x = torch.relu(x)
        # => [b x T x 88 x 1]
        x = self.linear_2(x)
        # x = torch.relu(x)
        # => [b x T x 88]
        x = torch.squeeze(x, dim=3)
        # => [b x T x 768]
        # x = self.linear3(x)
        # x = torch.relu(x)
        x = torch.sigmoid(x)
        
        return x