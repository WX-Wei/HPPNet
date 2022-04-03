"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
import torchaudio

from .lstm import BiLSTM
from .mel import melspectrogram

from .nets import HarmSpecgramConvBlock, HarmSpecgramConvNet, MRDConvNet

from .ChromaNet import ChromaNet

from .constants import *

import nnAudio


e = 2**(1/24)
to_log_specgram = nnAudio.Spectrogram.STFT(sr=SAMPLE_RATE, n_fft=WINDOW_LENGTH, freq_bins=88*4, hop_length=HOP_LENGTH, freq_scale='log', fmin=27.5/e, fmax=4186.0*e, output_format='Magnitude').to(DEFAULT_DEVICE)
# nnAudio.Spectrogram.CQT(sr=22050, hop_length=512, fmin=32.7, fmax=None, n_bins=84, bins_per_octave=12, filter_scale=1, norm=1, window='hann', center=True, pad_mode='reflect', trainable=False, output_format='Magnitude', verbose=True)
to_cqt = nnAudio.Spectrogram.CQT(sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmin=27.5/e, n_bins=88*4, bins_per_octave=BINS_PER_SEMITONE*12, output_format='Magnitude').to(DEFAULT_DEVICE)

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
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # if 'onset' in SUB_NETS:
        #     self.onset_stack = nn.Sequential(
        #         ConvStack(input_features, model_size),
        #         sequence_model(model_size, model_size),
        #         nn.Linear(model_size, output_features),
        #         nn.Sigmoid()
        #         # MRDConvNet(),
        #     )
        # if 'offset' in SUB_NETS:
        #     self.offset_stack = nn.Sequential(
        #         ConvStack(input_features, model_size),
        #         sequence_model(model_size, model_size),
        #         nn.Linear(model_size, output_features),
        #         nn.Sigmoid()
        #         # MRDConvNet(),
        #     )

        if 'frame' in SUB_NETS:
            self.frame_stack = nn.Sequential(
                # ConvStack(input_features, model_size),
                # HarmSpecgramConvBlock(88),
                # HarmSpecgramConvNet(88),

                MRDConvNet(),

                # ChromaNet(),

                # sequence_model(88 , 88),
                # nn.Linear(88, 88),
                # nn.ReLU()
            )
            self.combined_stack = nn.Sequential(
                sequence_model(4, 64),
                # sequence_model(8, 8),
                nn.Linear(64, 1),
                nn.Sigmoid(),
                Squeeze(2)
            )
        # if 'velocity' in SUB_NETS:
        #     self.velocity_stack = nn.Sequential(
        #         ConvStack(input_features, model_size),
        #         # HarmSpecgramConvBlock(model_size),
        #         nn.Linear(model_size, output_features)
        #     )

    def forward(self, waveforms):

        waveforms = waveforms.to(DEFAULT_DEVICE)


        # => [n_mel x T] => [T x n_mel]
        mel = melspectrogram(waveforms).transpose(-1, -2)[:, :self.frame_num, :]

        # => [b x T x 352]
        # log_gram_mag = to_log_specgram(waveforms).swapaxes(1, 2).float()[:, :640, :]
        cqt = to_cqt(waveforms).swapaxes(1, 2).float()[:, :self.frame_num, :]
        spgcgram_db = self.amplitude_to_db(cqt)

        activation_pred, onset_pred, offset_pred, velocity_pred = self.frame_stack(spgcgram_db)

        results = []
        if 'onset' in SUB_NETS:
            # onset_pred = self.onset_stack(mel)
            results.append(onset_pred)
        if 'offset' in SUB_NETS:
            # offset_pred = self.offset_stack(mel)
            results.append(offset_pred)
        if 'frame' in SUB_NETS:
            
            results.append(activation_pred)

        
            # combined_pred = torch.unsqueeze(activation_pred, 3)
            # if 'onset' in SUB_NETS:
            #     combined_pred = torch.cat([torch.unsqueeze(onset_pred, 3).detach(), combined_pred], dim=-1)
            # if 'offset' in SUB_NETS:
            #     combined_pred = torch.cat([torch.unsqueeze(offset_pred, 3).detach(), combined_pred], dim=-1)
            # if 'velocity' in SUB_NETS:
            #     # print(velocity_pred.size(), combined_pred.size())
            #     combined_pred = torch.cat([torch.unsqueeze(velocity_pred, 3).detach(), combined_pred], dim=-1)
            # combined_pred = torch.permute(combined_pred, [0, 2, 1, 3])
            # # => [(b*88) x T x 3]
            # combined_pred = combined_pred.reshape([-1, combined_pred.size()[2], combined_pred.size()[3]])
            # frame_pred = self.combined_stack(combined_pred)
            # frame_pred = torch.reshape(frame_pred, [-1, 88, frame_pred.size()[-1]])
            # frame_pred = torch.permute(frame_pred, [0, 2, 1])
            # results.append(frame_pred)
        if 'velocity' in SUB_NETS:
            # velocity_pred = self.velocity_stack(mel)
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
            losses['loss/onset'] = F.binary_cross_entropy(predictions['onset'], onset_label)
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

