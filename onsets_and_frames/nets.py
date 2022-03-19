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




from .constants import *

from .layers import WaveformToHarmgram

from .lstm import BiLSTM

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
        x = torch.relu(x)
        # x = torch.sigmoid(x)
        
        return x


class HarmSpecgramConvNet(nn.Module):
    def get_conv3d_block(self, channel_in,channel_out, kernel_size = [1, 3, 3], pool_size = [1, 2, 2], dilation = [1, 1, 1]):
        return nn.Sequential( 
            nn.Conv3d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.MaxPool3d(pool_size),
            nn.BatchNorm3d(channel_out),
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

        self.block_1 = self.get_conv3d_block(1, 16, dilation=[1, 48, 1])
        self.block_2 = self.get_conv3d_block(16, 32, dilation=[1, 24, 1])
        self.block_3 = self.get_conv3d_block(32, 64, pool_size=[1, 1, 2], dilation=[1, 12, 1])
        self.block_4 = self.get_conv3d_block(64, 64, pool_size=[1, 1, 1], dilation=[1, 12, 1])
        self.block_5 = self.get_conv3d_block(64, 64, kernel_size=[1,3,1], pool_size=[1, 1, 1], dilation=[1, 12, 1])
        self.block_6 = self.get_conv3d_block(64, 128, kernel_size=[1,3,1], pool_size=[1, 1, 1], dilation=[1, 12, 1])

        self.lstm = BiLSTM(128, 32)
        self.linear_rnn = nn.Linear(64, 1)


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

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        # => [b x ch x T x 88 x 1]
        x = self.block_6(x)
        # => [b x ch x T x 88]
        x = torch.squeeze(x, dim = 4)


        # # => [b x T x 88 x ch]
        # x = torch.permute(x, [0, 2, 3, 1])

        # x = self.linear_1(x)
        # x = torch.relu(x)
        # # => [b x T x 88 x 1]
        # x = self.linear_2(x)
        # # x = torch.relu(x)
        # # => [b x T x 88]
        # x = torch.squeeze(x, dim=3)
        # # => [b x T x 768]
        # # x = self.linear3(x)
        # # x = torch.relu(x)
        # x = torch.sigmoid(x)

        # => [b x 88 x T x ch]
        x = torch.swapdims(x, 1, 3)
        s = x.size()
        # => [(b*88) x T x ch]
        x = x.reshape(s[0]*s[1], s[2], s[3])
        # => [(b*88) x T x 64]
        x = self.lstm(x)
        # => [(b*88) x T x 1]
        x = self.linear_rnn(x)
        x = torch.sigmoid(x)
        # => [b x 88 x T]
        x = x.reshape(s[0], s[1], s[2])
        # => [b x T x 88]
        x = torch.swapdims(x, 1, 2)

        y = x.detach()
        y 

        
        return x


class TimeChannelWiseLSTM(nn.Module):
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



class MRDConvNet(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = [1, 1], dilation = [1, 1]):
        return nn.Sequential( 
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.BatchNorm2d(channel_out),
        )

    def __init__(self) -> None:
        super().__init__()
        self.device = DEFAULT_DEVICE


        self.block_1 = self.get_conv2d_block(2, 16, kernel_size=3)
        self.block_2 = self.get_conv2d_block(16, 16, kernel_size=3)

        # self.block_3 = MRDC_Conv(16, 64, dilation_list=[48, 76, 96, 111, 124, 135, 144, 152, 159, 166])

        self.conv_3_1 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 48])
        self.conv_3_2 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 76])
        self.conv_3_3 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 96])
        self.conv_3_4 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 111])
        self.conv_3_5 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 124])
        self.conv_3_6 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 135])
        self.conv_3_7 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 144])
        self.conv_3_8 = nn.Conv2d(16, 64, [1, 3], padding='same', dilation=[1, 152])

        self.block_4 = self.get_conv2d_block(64, 128, pool_size=[1, 2], dilation=[1, 24])
        self.block_5 = self.get_conv2d_block(128, 128, pool_size=[1, 2], dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(128, 128, dilation=[1, 12])

        lstm_size = 64

        self.lstm = BiLSTM(128, lstm_size//2)
        self.lstm_onsets = BiLSTM(128, lstm_size//2)
        self.lstm_offsets = BiLSTM(128, lstm_size//2)
        self.lstm_velocity = BiLSTM(128, lstm_size//2)

        self.linear_rnn = nn.Linear(lstm_size, 1)
        self.linear_rnn_onsets = nn.Linear(lstm_size, 1)
        self.linear_rnn_offsets = nn.Linear(lstm_size, 1)

        self.TCW_lstm_frame_low = TimeChannelWiseLSTM(128, 1, 64)
        self.TCW_lstm_frame_mid = TimeChannelWiseLSTM(128, 1, 64)
        self.TCW_lstm_frame_high = TimeChannelWiseLSTM(128, 1, 64)

        self.conv_velocity = nn.Conv2d(128, 1, 1)

        


    def forward(self, log_gram_db):
        # inputs: [b x T x n_freq]
        # outputs: [b x T x 88]


        img_path = 'logspecgram_preview.png'
        if not os.path.exists(img_path):
            img = torch.permute(log_gram_db, [2, 0, 1]).reshape([352, 640*4]).detach().cpu().numpy()
            # x_grid = torchvision.utils.make_grid(x.swapaxes(0, 1), pad_value=1.0).swapaxes(0, 2).detach().cpu().numpy()
            # plt.imsave(img_path, (x_grid+80)/100)
            plt.imsave(img_path, img)

            time_wise_diff = img[:, 1:] - img[:, :-1]
            plt.imsave('logspecgram_diff_preview.png', time_wise_diff)

        log_gram_db_delta = log_gram_db + 0
        log_gram_db_delta[:, 1:, :] = log_gram_db_delta[:, 1:, :] - log_gram_db_delta[:, :-1, :]
        log_gram_db_delta = torch.unsqueeze(log_gram_db_delta, dim=1)


        # => [b x 1 x T x 352]
        x = torch.unsqueeze(log_gram_db, dim=1)
        # => [b x 2 x T x 352]
        x = torch.cat([x, log_gram_db_delta], dim=1)

        x = self.block_1(x)
        x = self.block_2(x)

        x = self.conv_3_1(x) + self.conv_3_2(x) + self.conv_3_3(x) + self.conv_3_4(x) + self.conv_3_5(x) + self.conv_3_6(x) + self.conv_3_7(x) + self.conv_3_8(x)
        # x = self.block_3(x)
        x = torch.relu(x)

        x = self.block_4(x)
        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x)

        # => [b x 1 x T x 88]
        x_velocity = self.conv_velocity(x)
        x_velocity = torch.relu(x_velocity)
        # => [b x T x 88]
        x_velocity = torch.unsqueeze(x_velocity, dim=1)


        # x_onset = self.TCW_lstm_onset(x)
        # x_onset = torch.squeeze(x_onset, dim=1)

        # x_offset = self.TCW_lstm_offset(x)
        # x_offset = torch.squeeze(x_offset, dim=1)

        x_frame_low = self.TCW_lstm_frame_low(x[:, :, :, :29])
        x_frame_mid = self.TCW_lstm_frame_mid(x[:, :, :, 29:59])
        x_frame_high = self.TCW_lstm_frame_high(x[:, :, :, 59:])
        x_frame = torch.cat([x_frame_low, x_frame_mid, x_frame_high], dim=3)
        x_frame = torch.squeeze(x_frame, dim=1)

        # x_frame = torch.squeeze(x_frame, dim=1)

        # => [b x 88 x T x ch]
        x = torch.swapdims(x, 1, 3)
        s = x.size()
        # => [(b*88) x T x ch]
        x = x.reshape(s[0]*s[1], s[2], s[3])

        x_0 = x


        # # => [(b*88) x T x 64]
        # x = self.lstm(x)
        # # => [(b*88) x T x 1]
        # x = self.linear_rnn(x)
        # x = torch.sigmoid(x)
        # # => [b x 88 x T]
        # x = x.reshape(s[0], s[1], s[2])
        # # => [b x T x 88]
        # x_frame = torch.swapdims(x, 1, 2)

        # => [(b*88) x T x 64]
        x_onset = self.lstm_onsets(x_0)
        # => [(b*88) x T x 1]
        x_onset = self.linear_rnn_onsets(x_onset)
        x_onset = torch.sigmoid(x_onset)
        # => [b x 88 x T]
        x_onset = x_onset.reshape(s[0], s[1], s[2])
        # => [b x T x 88]
        x_onset = torch.swapdims(x_onset, 1, 2)

        # => [(b*88) x T x 64]
        x_offset = self.lstm_offsets(x_0)
        # => [(b*88) x T x 1]
        x_offset = self.linear_rnn_offsets(x_offset)
        x_offset = torch.sigmoid(x_offset)
        # => [b x 88 x T]
        x_offset = x_offset.reshape(s[0], s[1], s[2])
        # => [b x T x 88]
        x_offset = torch.swapdims(x_offset, 1, 2)
        
        return x_frame, x_onset, x_offset, x_velocity