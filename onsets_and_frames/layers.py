import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
from torchaudio.transforms import Spectrogram
import nnAudio.Spectrogram


# 
# 12.        , 19.01955001, 24.        , 27.86313714,
#       31.01955001, 33.68825906, 36
class FlexibleDilationConv(nn.Module):
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


class CausalDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,3], dilation=[1, 1]) -> None:
        super().__init__()
        right = (kernel_size[1]-1) * dilation[1]
        bottom = (kernel_size[0]-1) * dilation[0]
        self.padding = nn.ZeroPad2d([0, right, 0 , bottom])
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self,x):
        x = self.padding(x)
        x = self.conv2d(x)
        return x


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.deptwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels )

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        #
        y = self.deptwise(x)
        y = self.pointwise(y)
        return y


# harmonicPower High Frequency Resolution
class HarmgramLogscale(nn.Module):
    def __init__(self, note_id, bins_per_semitone,  sample_rate, n_freq, n_har, device):
        super().__init__()

        self.note_id = note_id
        self.sample_rate = sample_rate
        self.n_freq = n_freq
        self.n_har = n_har
        self.bins_per_semitone = bins_per_semitone

        fre_resolution = sample_rate/(n_freq*2)
        base = np.power(2, 1/12.0)

        fre_max = n_freq * fre_resolution # max freq of specgram

        hargram_idx = np.ones([bins_per_semitone, n_har], dtype=np.int) * (n_freq - 1) # 

        mask_list = []
        for i_note in range(bins_per_semitone):
            f0 = 27.5 * (base ** (note_id - 0.5 + 0.5/bins_per_semitone + i_note/bins_per_semitone))

            n_har_true = min(n_har, int(np.floor(fre_max / f0)))
            for i_har in range(n_har_true):
                f_i = f0 * (i_har + 1)
                idx = int( f_i // fre_resolution )
                hargram_idx[i_note][i_har] = idx
        # => [bins_per_semitone x n_har]
        self.hargram_idx = torch.tensor(hargram_idx).to(device)

    def forward(self, specgram):
        # inputs: [b x T x (n_freq + 1)], the 0-th element of (n_freq+1) should be 0
        # outputs: [b x T x bins_per_semitone x har_n], [b x T x bins_per_semitone]

        harmgram = specgram[:, :, self.hargram_idx]
        return harmgram


class HarmgramLogscaleList(nn.Module):
    def __init__(self, sample_rate, n_freq, device, bins_per_semitone = 16, n_har = 15):
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.n_har = n_har
        lst = []
        for i in range(88):
            lst += [HarmgramLogscale(i, bins_per_semitone,  sample_rate, n_freq, n_har, device)]

        self.harmgram_logscale_lst = nn.ModuleList(lst)
        
    def forward(self, specgram):
        # inputs: [b x T x n_fre]
        # outputs:  [b x T x (88*bins_per_semitone) x n_har]

        b, T, n_fre = specgram.size()

        harmgram_lst = []
        for i in range(88):
            harmgram = self.harmgram_logscale_lst[i](specgram)
            harmgram_lst += [harmgram]

        # => [b x T x (88*bins_per_semitone) x n_har]
        harmgram_lst = torch.cat(harmgram_lst, dim=2)

        # => [b x T x 88]
        # lst_88 = nn.MaxPool1d(self.bins_per_semitone)(lst)

        return harmgram_lst # lst_88, lst


class WaveformToHarmgram(nn.Module):
    def __init__(self, sample_rate, n_freq, n_har, bins_per_semitone, hop_length, device):
        super().__init__()
        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_freq*2, hop_length=hop_length).to(device)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.get_harmgram_list = HarmgramLogscaleList(sample_rate, n_freq, device, n_har = n_har, bins_per_semitone=bins_per_semitone)

    def forward(self, waveforms):
        # output: [b x T x (88*bins_per_semitone) x n_har]
        # waveforms = torch.tensor(waveforms).to(device)
        # => [b x (n_fre+1) x (T+1)]
        specgram = self.waveform_to_specgram(waveforms)
        specgram = specgram[:, :, :-1] # 
        # => [b x T x n_fre]
        specgram = specgram.permute([0, 2, 1])

        #######################################
        # Calc DB
        # => [b x T x (88 * bins_per_semitone) x n_har], [b x T x (88*bins_per_semitone)]
        harmgram_list = self.get_harmgram_list(specgram)
        harmgram_list = self.amplitude_to_db(harmgram_list)
        # normalize
        # max_val = torch.max(harmgram_list)
        # min_val = torch.min(harmgram_list)
        # harmgram_list = (harmgram_list - min_val) / (max_val - min_val)
        return harmgram_list


class WaveformToLogSpecgram(nn.Module):
    def __init__(self, sample_rate, n_fft, fmin, bins_per_octave, freq_bins, hop_length, logspecgram_type, device):
        super().__init__()

        e = freq_bins/bins_per_octave
        fmax = fmin * (2 ** e)

        self.logspecgram_type = logspecgram_type
        self.n_fft = n_fft
        self.hamming_window = torch.hann_window(self.n_fft).to(device)
        self.hamming_window = torch.unsqueeze(self.hamming_window, 0)

        # torch.hann_window()

        fre_resolution = sample_rate/n_fft

        idxs = torch.arange(0, freq_bins, device=device)

        log_idxs = fmin * (2**(idxs/bins_per_octave)) / fre_resolution

        # 线性插值： y_k = y_i * (k-i) + y_{i+1} * ((i+1)-k)
        self.log_idxs_floor = torch.floor(log_idxs).long()
        self.log_idxs_floor_w = (log_idxs - self.log_idxs_floor).reshape([1, freq_bins, 1])
        self.log_idxs_ceiling = torch.ceil(log_idxs).long()
        self.log_idxs_ceiling_w = (self.log_idxs_ceiling - log_idxs).reshape([1, freq_bins, 1])

        self.waveform_to_specgram = torchaudio.transforms.Spectrogram(n_fft, hop_length=hop_length).to(device)

        assert(bins_per_octave % 12 == 0)
        bins_per_semitone = bins_per_octave // 12
        # self.waveform_to_harmgram = WaveformToHarmgram(sample_rate, n_freq=n_fft//2, n_har=1, bins_per_semitone=bins_per_semitone, hop_length=hop_length, device=device)
        
        if(logspecgram_type == 'logspecgram'):
            self.spec_layer = nnAudio.Spectrogram.STFT(
                n_fft=n_fft, 
                freq_bins=freq_bins, 
                hop_length=hop_length, 
                sr=sample_rate,
                freq_scale='log',
                fmin=fmin,
                fmax=fmax,
                output_format='Magnitude'
            ).to(device)
        elif(logspecgram_type == 'cqt'):
            self.spec_layer = nnAudio.Spectrogram.CQT(
                sr=sample_rate, hop_length=hop_length,
                fmin=fmin,
                n_bins=freq_bins,
                bins_per_octave=bins_per_octave,
                output_format='Magnitude',
            ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveforms):
        # inputs: [b x wave_len]
        # outputs: [b x T x n_bins]


        if(self.logspecgram_type == 'logharmgram'):
            # return self.waveform_to_harmgram(waveforms)[:, :, :, 0]

            # [b x (n_fft/2 + 1) x T]
            # specgram = self.waveform_to_specgram(waveforms)

            waveforms = waveforms * self.hamming_window
            specgram =  torch.fft.fft(waveforms)
            specgram = torch.abs(specgram[:, :self.n_fft//2 + 1])
            specgram = specgram * specgram
            specgram = torch.unsqueeze(specgram, dim=2)

            # => [b x freq_bins x T]
            specgram = specgram[:, self.log_idxs_floor] * self.log_idxs_floor_w + specgram[:, self.log_idxs_ceiling] * self.log_idxs_ceiling_w
        elif(self.logspecgram_type == 'cqt' or self.logspecgram_type == 'logspecgram'):
            specgram = self.spec_layer(waveforms)

        specgram_db = self.amplitude_to_db(specgram)
        # specgram_db = (specgram_db + 80)/80
        # specgram_db = specgram_db[:, :, :-1] # remove the last frame.
        specgram_db = specgram_db.permute([0, 2, 1])
        return specgram_db




