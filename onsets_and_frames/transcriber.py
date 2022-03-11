"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram

from .nets import HarmSpecgramConvBlock, HarmSpecgramConvNet, MRCDConvNet

from .constants import *


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

        if 'onset' in SUB_NETS:
            self.onset_stack = nn.Sequential(
                # ConvStack(input_features, model_size),
                HarmSpecgramConvBlock(model_size),
                sequence_model(model_size, model_size),
                nn.Linear(model_size, output_features),
                nn.Sigmoid()
            )
        if 'offset' in SUB_NETS:
            self.offset_stack = nn.Sequential(
                # ConvStack(input_features, model_size),
                HarmSpecgramConvBlock(model_size),
                sequence_model(model_size, model_size),
                nn.Linear(model_size, output_features),
                nn.Sigmoid()
            )

        if 'frame' in SUB_NETS:
            self.frame_stack = nn.Sequential(
                # ConvStack(input_features, model_size),
                # HarmSpecgramConvBlock(88),
                # HarmSpecgramConvNet(88),
                MRCDConvNet(),
                # sequence_model(88 , 88),
                # nn.Linear(88, 88),
                # nn.ReLU()
            )
            self.combined_stack = nn.Sequential(
                # sequence_model(output_features * 2, model_size),
                # nn.Linear(output_features, output_features),
                # nn.Sigmoid()
            )
        if 'velocity' in SUB_NETS:
            self.velocity_stack = nn.Sequential(
                # ConvStack(input_features, model_size),
                HarmSpecgramConvBlock(48*16),
                nn.Linear(48*16, output_features)
            )

    def forward(self, mel):
        results = []
        if 'onset' in SUB_NETS:
            onset_pred = self.onset_stack(mel)
            results.append(onset_pred)
        if 'offset' in SUB_NETS:
            offset_pred = self.offset_stack(mel)
            results.append(offset_pred)
        if 'frame' in SUB_NETS:
            activation_pred = self.frame_stack(mel)
            results.append(activation_pred)
            # combined_pred = activation_pred

            # if 'onset' in SUB_NETS:
            #     combined_pred = torch.cat([onset_pred.detach(), combined_pred], dim=-1)
            # if 'offset' in SUB_NETS:
            #     combined_pred = torch.cat([offset_pred.detach(), combined_pred], dim=-1)
            # frame_pred = self.combined_stack(combined_pred)
            # results.append(frame_pred)
        if 'velocity' in SUB_NETS:
            velocity_pred = self.velocity_stack(mel)
            results.append(velocity_pred)
        # return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
        return results

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        audio_label_reshape = audio_label.reshape(-1, audio_label.shape[-1])#[:, :-1]
        # => [n_mel x T] => [T x n_mel]
        # mel = melspectrogram(audio_label_reshape).transpose(-1, -2)

        # => [T x (88*4) x 16]


        mel = audio_label_reshape

        # => [T x 88]
        # onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        results = self(mel)
        idx = 0
        predictions = {
            'onset': onset_label,
            'offset': offset_label,
            'frame': frame_label,
            'velocity': velocity_label
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
            losses['loss/frame'] = - 10 * y_ref * torch.log(y_pred) - (1-y_ref)*torch.log(1-y_pred)  # F.binary_cross_entropy(predictions['frame'], frame_label)
            losses['loss/frame'] = losses['loss/frame'].mean()
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

