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

from .nets import  CNNTrunk, FrqeBinLSTM

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

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)

class Head(nn.Module):
    def __init__(self, head_type, model_size) -> None:
        super().__init__()
        heads = {
            'FB-LSTM': FrqeBinLSTM(model_size, 1, model_size) ,
            'Conv': nn.Sequential(nn.Conv2d(model_size, 1, 1), nn.Sigmoid()),
            # [B x model_size x T x 88]
            'LSTM':nn.Sequential(
                nn.Conv2d(model_size, 1, 1), # => [B x 1 x T x 88]
                nn.ReLU(),
                Squeeze(dim=1), # => [B x T x 88]
                BiLSTM(88,model_size//2), # => [B x T x model_size]
                nn.Linear(model_size, 88), # => [B x T x 88]
                nn.Sigmoid(),
                Unsqueeze(dim=1) # =>[B x 1 x T x 88]
            )
        }
        self.head = heads[head_type]
    def forward(self, x):
        # input: [B x model_size x T x 88]
        # output: [B x 1 x T x 88]
        y = self.head(x)
        # y = torch.squeeze(y, dim=1)
        return y

class SubNet(nn.Module):
    def __init__(self, model_size = 128, trunk_type = "HD-Conv", head_type = "FB-LSTM",  head_names = ['head'], concat = False, time_pooling = False) -> None:
        super().__init__()
        # Trunk
        self.trunk = CNNTrunk(c_in=2, c_har=16, embedding=model_size, trunk_type=trunk_type)

        # Heads
        head_size = model_size
        self.concat = concat
        if(concat):
            head_size *= 2
        self.head_names = head_names
        self.heads = nn.ModuleDict()
        for name in head_names:
            self.heads[name] = Head(head_type, head_size)

        self.time_pooling = time_pooling
       
    def forward(self, x, piano_roll_mask):
        # input: [B x 2 x T x 352], [B x 1 x T x 88]
        # output:
        #   {"head_1": [B x T x 88], 
        #    "head_2": [B x T x 88],...
        # }
        
        piano_roll_mask_new = piano_roll_mask
        if(self.time_pooling):
            x = F.max_pool2d(x, [2,1])
            src_size = piano_roll_mask.size()
            piano_roll_mask_new = F.max_pool2d(piano_roll_mask, [2,1])

        # => [B x model_size x T x 88]
        y = self.trunk(x, piano_roll_mask_new)

        output = {}
        for head in self.head_names:
            # => [B x 1 x T x 88]
            output[head] = self.heads[head](y)
            if(self.time_pooling):
                output[head] = F.upsample(output[head], size=src_size[-2:], mode='bilinear')
            
            output[head] = output[head] * piano_roll_mask
            # output[head] = torch.squeeze(output[head], dim=1)
            output[head] = torch.clip(output[head], 1e-7, 1-1e-7)

        return output

class HARPIST(nn.Module):
    def __init__(self, input_features, output_features, config):
        super().__init__()
        self.config = config
        model_size = config['model_size']
        head_type = config['head_type']
        trunk_type = config['trunk_type']

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.subnet_onset = SubNet(model_size, trunk_type, head_type, config['onset_subnet_heads'])
        self.subnet_frame = SubNet(model_size, trunk_type, head_type, config['frame_subnet_heads'], time_pooling=True)
        self.combined_FBLSTM = FrqeBinLSTM(3, 1, 128)


        self.sub_nets = {}
        self.sub_nets['onset_subnet'] = self.subnet_onset
        self.sub_nets['frame_subnet'] = nn.ModuleList([self.subnet_frame, self.combined_FBLSTM])
        self.sub_nets['all'] = nn.ModuleList([self.subnet_onset, self.subnet_frame])

    def forward(self, waveforms, piano_roll_mask):
        # inputs: [b x n], [b x T x 88]
        # outputs: 
        '''
        {
            "onset":[b x T x 88],
            "frame":
            "offset":
            "velocity":
            "combined_frame":
        }
        '''

        waveforms = waveforms.to(self.config['device'])
        piano_roll_mask = piano_roll_mask.to(self.config['device'])
        piano_roll_mask = torch.unsqueeze(piano_roll_mask, dim=1)
        global to_cqt
        global to_log_specgram
        to_cqt = to_cqt.to(self.config['device'])
        to_log_specgram = to_log_specgram.to(self.config['device'])

        # => [n_mel x T] => [T x n_mel]
        # mel = melspectrogram(waveforms).transpose(-1, -2)[:, :self.frame_num, :]

        # => [b x T x 352]
        # log_gram_mag = to_log_specgram(waveforms).swapaxes(1, 2).float()[:, :640, :]
        cqt = to_cqt(waveforms).swapaxes(1, 2).float()
        log_specgram = to_log_specgram(waveforms).swapaxes(1, 2).float()
        # log_specgram_2 = to_log_specgram(waveforms).swapaxes(1, 2).float()[:, :self.frame_num, :]
        
        cqt_db = self.amplitude_to_db(cqt)
        log_specgram_db = self.amplitude_to_db(log_specgram)
        # log_specgram_db_2 = self.amplitude_to_db(log_specgram_2)

        # => [b x 2 x T x 352]
        specgram_db = torch.stack([log_specgram_db, cqt_db], dim=1)
        specgram_db = specgram_db[:, :, :self.frame_num, :]
        pad_len = self.frame_num - specgram_db.size()[2]
        if(pad_len > 0):
            print(f'frame len < {self.frame_num}, zero_pad_len:{pad_len}')
            # => [B x 2 x T x 352]
            specgram_db = F.pad(specgram_db, [0, 0, 0, pad_len], mode='replicate')
            assert specgram_db.size()[2] == self.frame_num
        # specgram_db = cqt_db

        results = self.subnet_onset(specgram_db, piano_roll_mask.detach())
        results_2 = self.subnet_frame(specgram_db, piano_roll_mask.detach())
        results.update(results_2)

        onset_high_conf = torch.max(results['onset'], (results['onset'] >= 0.4).float())
        offset_high_conf = torch.max(results['offset'], (results['offset'] >= 0.4).float())

        # => [b x 3 x T x 88]
        combined_frame = torch.concat([results['frame'], onset_high_conf.detach(), offset_high_conf.detach()], dim=1)
        results['combined_frame'] = self.combined_FBLSTM(combined_frame)
        return results

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        self.frame_num = frame_label.size()[-2]
        self.piano_roll_size = frame_label.size()[-2:]

        audio_label_reshape = audio_label.reshape(-1, audio_label.shape[-1])#[:, :-1]
        # => [n_mel x T] => [T x n_mel]
        # mel = melspectrogram(audio_label_reshape).transpose(-1, -2)

        onset_amend = onset_label + 0
        frame_tmp = frame_label + 0
        if(len(onset_amend.size()) == 2):
            # => [1 x T x 88]
            onset_amend = torch.unsqueeze(onset_amend, dim=0)
            frame_tmp = torch.unsqueeze(frame_tmp, dim=0)
        onset_amend[:, 0, :] = onset_amend[:, 0, :] + frame_tmp[:, 0, :]
        onset_amend = torch.clip(onset_amend, 0, 1)

        onset_pad = F.pad(onset_amend, [0, 0, 999, 0])
        piano_roll_mask = F.max_pool2d(onset_pad, [1000, 1], stride=[1, 1])
        piano_roll_mask = piano_roll_mask[:, :self.frame_num, :]

        # don't use mask
        piano_roll_mask = piano_roll_mask - piano_roll_mask + 1


        # => [T x 88]
        # onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        results = self.forward(audio_label_reshape, piano_roll_mask)
        predictions = {
            'onset': torch.clip(onset_label, 0, 0),
            'offset': torch.clip(offset_label, 0, 0),
            'frame': torch.clip(frame_label, 0, 0),
            'velocity': torch.clip(velocity_label, 0, 0)
        }
        losses = {}
        bce = lambda x, y: -y *torch.log(x) - (1-y)*torch.log(1-x)
        if 'onset' in results.keys():
            predictions['onset'] = results['onset'].reshape(*onset_label.shape)
            # [b x T x 88]
            losses['loss/onset'] = - 2 * onset_label * torch.log(predictions['onset']) - ( 1 - onset_label) * torch.log(1-predictions['onset'])
            losses['loss/onset'] = losses['loss/onset'].mean()
            # losses['loss/onset'] = F.binary_cross_entropy(predictions['onset'], onset_label)
        if 'offset' in results.keys():
            predictions['offset'] = results['offset'].reshape(*offset_label.shape)
            losses['loss/offset'] = F.binary_cross_entropy(predictions['offset'], offset_label)
        if 'frame' in results.keys():
            predictions['frame'] = results['combined_frame'].reshape(*frame_label.shape)
            activation = results['frame'].reshape(*frame_label.shape)
            losses['loss/frame'] = F.binary_cross_entropy(predictions['frame'], frame_label) + \
                                    F.binary_cross_entropy(activation, frame_label)
            # losses['loss/frame'] = bce(predictions['frame'] , frame_label) * piano_roll_mask
            # mask_sum = (piano_roll_mask.sum()+0.0000001)
            # losses['loss/frame'] = (losses['loss/frame']/mask_sum).sum()
        if 'velocity' in results.keys():
            predictions['velocity'] = results['velocity'].reshape(*velocity_label.shape)
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        losses['loss/all'] = sum(losses.values())

        losses['loss/onset_subnet'] = torch.tensor(0.0).to(self.config['device'])
        for head in self.config['onset_subnet_heads']:
            losses['loss/onset_subnet'] += losses['loss/' + head]

        losses['loss/frame_subnet'] = torch.tensor(0.0).to(self.config['device'])
        for head in self.config['frame_subnet_heads']:
            losses['loss/frame_subnet'] += losses['loss/' + head]

        

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

