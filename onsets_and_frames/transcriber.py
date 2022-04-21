"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import lstm, nn
import torchaudio

from .lstm import BiLSTM
from .mel import melspectrogram

from .nets import  HarmonicDilatedConv, FrqeBinLSTM

from .constants import *

import nnAudio


e = 2**(1/24)
to_log_specgram = nnAudio.Spectrogram.STFT(sr=SAMPLE_RATE, n_fft=512, freq_bins=88*4, hop_length=HOP_LENGTH, freq_scale='log', fmin=27.5/e, fmax=4186.0*e, output_format='Magnitude')
# to_log_specgram_2 = nnAudio.Spectrogram.STFT(sr=SAMPLE_RATE, n_fft=2048, freq_bins=88*4, hop_length=HOP_LENGTH, freq_scale='log', fmin=27.5/e, fmax=4186.0*e, output_format='Magnitude')
# nnAudio.Spectrogram.CQT(sr=22050, hop_length=512, fmin=32.7, fmax=None, n_bins=84, bins_per_octave=12, filter_scale=1, norm=1, window='hann', center=True, pad_mode='reflect', trainable=False, output_format='Magnitude', verbose=True)
to_cqt = nnAudio.Spectrogram.CQT(sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=27.5/e, n_bins=88*4, bins_per_octave=BINS_PER_SEMITONE*12, output_format='Magnitude')

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.squeeze(x, self.dim)

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        # add channel dimention
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        # =>
        x = self.cnn(x)
        # [C x T x F] => [T x C x F] => [T x (C*F)]
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x




class OnsetsAndFrames(nn.Module):
    def get_head(self, type, model_size):
        heads = {
            'FB-LSTM': FrqeBinLSTM(model_size, 1, model_size) ,
            'Conv': nn.Sequential(nn.Conv2d(model_size, 1, 1), nn.Sigmoid())
        }
        return heads[type]
    def __init__(self, input_features, output_features, config):
        super().__init__()

        self.config = config

        model_size = config['model_size']
        head_type = config['head_type']
        trunk_type = config['trunk_type']
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        global to_cqt
        global to_log_specgram
        to_cqt = to_cqt.to(config['device'])
        to_log_specgram = to_log_specgram.to(config['device'])
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.sub_nets = {}

        if 'onset' in SUB_NETS:
            self.onset_dict = nn.ModuleDict({
                'onset_trunk': HarmonicDilatedConv(c_in=2, c_har=16, embedding=model_size),
                'onset_head': self.get_head(head_type, model_size)
            })
            self.sub_nets['onset'] = self.onset_dict
        if 'frame' in SUB_NETS:
            self.frame_dict = nn.ModuleDict({
                'frame_trunk': HarmonicDilatedConv(c_in=2, c_har=16, embedding=model_size),
                'frame_head': self.get_head(head_type, model_size*2)
            })
            self.sub_nets['frame'] = self.frame_dict
        if 'velocity' in SUB_NETS:
            self.velocity_dict = nn.ModuleDict({
                'velocity_trunk': HarmonicDilatedConv(c_in=2, c_har=4, embedding=4),
                'velocity_head': self.get_head(head_type, 4)
            })
            self.sub_nets['velocity'] = self.velocity_dict

    def forward(self, waveforms):

        waveforms = waveforms.to(DEFAULT_DEVICE)


        # => [n_mel x T] => [T x n_mel]
        # mel = melspectrogram(waveforms).transpose(-1, -2)[:, :self.frame_num, :]

        # => [b x T x 352]
        # log_gram_mag = to_log_specgram(waveforms).swapaxes(1, 2).float()[:, :640, :]
        cqt = to_cqt(waveforms).swapaxes(1, 2).float()[:, :self.frame_num, :]
        log_specgram = to_log_specgram(waveforms).swapaxes(1, 2).float()[:, :self.frame_num, :]
        # log_specgram_2 = to_log_specgram(waveforms).swapaxes(1, 2).float()[:, :self.frame_num, :]
        
        cqt_db = self.amplitude_to_db(cqt)
        log_specgram_db = self.amplitude_to_db(log_specgram)
        # log_specgram_db_2 = self.amplitude_to_db(log_specgram_2)

        # => [b x 2 x T x 352]
        specgram_db = torch.stack([log_specgram_db, cqt_db], dim=1)
        # specgram_db = cqt_db

        # activation_pred, onset_pred, offset_pred, velocity_pred = self.frame_stack(specgram_db)

        results = []
        if 'onset' in SUB_NETS:
            # onset_pred = self.onset_stack(mel)
            onset_embeding = self.onset_dict['onset_trunk'](specgram_db)
            onset_pred = self.onset_dict['onset_head'](onset_embeding)
            onset_pred = torch.clip(onset_pred, 1e-7, 1 - 1e-7)
            results.append(onset_pred)

        # down sampling time dim, frames pred don't need high time resolution.
        specgram_db_pool = F.avg_pool2d(specgram_db, [2,1])
        if 'frame' in SUB_NETS:
            frame_embeding = self.frame_dict['frame_trunk'](specgram_db_pool)
            # => [B x (c_onset+c_frame) x T x 88]
            onset_embeding_pool = F.max_pool2d(onset_embeding, [2,1])
            stack_embeding = torch.cat([onset_embeding_pool.detach(), frame_embeding], dim=1)
            frame_pred = self.frame_dict['frame_head'](stack_embeding)
            # frame_pred = torch.squeeze(frame_pred, dim=1)
            # [B x 1 x T/2 x 88] => [B x 1 x T x 88]
            # frame_pred = torch.repeat_interleave(frame_pred, 2, dim=2)
            frame_pred = F.upsample(frame_pred, scale_factor=[2,1], mode='bilinear')
            results.append(frame_pred)

        if 'velocity' in SUB_NETS:
            # velocity_pred = self.velocity_stack(mel)
            velocity_embeding = self.velocity_dict['velocity_trunk'](specgram_db_pool)
            velocity_pred = self.velocity_dict['velocity_head'](velocity_embeding)
            # velocity_pred = torch.squeeze(velocity_pred, dim=1)
            velocity_pred = F.upsample(velocity_pred, scale_factor=[2,1], mode='bilinear')
            results.append(velocity_pred)
        # return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
        return results

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        self.frame_num = frame_label.size()[-2]

        audio_label_reshape = audio_label.reshape(-1, audio_label.shape[-1])#[:, :-1]
        # => [n_mel x T] => [T x n_mel]
        # mel = melspectrogram(audio_label_reshape).transpose(-1, -2)

        # => [T x (88*4) x 16]

        # => [T x 88]
        # onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        results = self(audio_label_reshape)
        idx = 0
        predictions = {
            'onset': torch.clip(onset_label, 0, 0),
            'offset': torch.clip(offset_label, 0, 0),
            'frame': torch.clip(frame_label, 0, 0),
            'velocity': torch.clip(velocity_label, 0, 0)
        }
        losses = {}
        if 'onset' in SUB_NETS:
            predictions['onset'] = results[idx].reshape(*onset_label.shape)
            # [b x T x 88]
            onset_ref_soft = onset_label.clone()
            losses['loss/onset'] = - 2 * onset_label * torch.log(predictions['onset']) - ( 1 - onset_label) * torch.log(1-predictions['onset'])
            losses['loss/onset'] = losses['loss/onset'].mean()
            # losses['loss/onset'] = F.binary_cross_entropy(predictions['onset'], onset_label)
            idx += 1
        if 'offset' in SUB_NETS:
            predictions['offset'] = results[idx].reshape(*offset_label.shape)
            losses['loss/offset'] = F.binary_cross_entropy(predictions['offset'], offset_label)
            idx += 1
        if 'frame' in SUB_NETS:
            predictions['frame'] = results[idx].reshape(*frame_label.shape)
            y_pred = torch.clip(predictions['frame'], 1e-4, 1 - 1e-4)
            y_ref = frame_label 
            # losses['loss/frame'] = - 10 * y_ref * torch.log(y_pred) - (1-y_ref)*torch.log(1-y_pred)  # F.binary_cross_entropy(predictions['frame'], frame_label)
            # losses['loss/frame'] = losses['loss/frame'].mean()
            losses['loss/frame'] = F.binary_cross_entropy(predictions['frame'], frame_label)
            idx += 1
        if 'velocity' in SUB_NETS:
            predictions['velocity'] = results[idx].reshape(*velocity_label.shape)
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        # onset_pred,  _, frame_pred, velocity_pred = self(mel)

        # predictions = {
        #     'onset': onset_pred.reshape(*onset_label.shape),
        #     # 'offset': offset_pred.reshape(*offset_label.shape),
        #     'frame': frame_pred.reshape(*frame_label.shape),
        #     'velocity': velocity_pred.reshape(*velocity_label.shape)
        # }

        # losses = {
        #     'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        #     # 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
        #     'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
        #     'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

