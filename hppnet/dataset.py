import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

import torch.utils.data


import h5py
import librosa
import pandas as pd

from .constants import *
from .midi import parse_midi



os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"

def max_pooling(x, pooling_size):
    # => [1 x W x H]
    x = torch.unsqueeze(x, dim=0)
    x = torch.max_pool2d(x, pooling_size)
    # => [W x H]
    x = torch.squeeze(x, dim=0)
    return x



class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        '''
        # reutrn :
        # [n_step, midi_bins]
        '''

        h5_path = self.data[index]
        with h5py.File(h5_path, 'r') as data:
            result = dict(path=data['path'][()])

            if self.sequence_length is not None:
                audio_length = data['audio'].shape[0] # len(data['audio'])
                step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
                n_steps = self.sequence_length // HOP_LENGTH
                step_end = step_begin + n_steps

                begin = step_begin * HOP_LENGTH
                end = begin + self.sequence_length

                result['audio'] = data['audio'][begin:end]
                result['label'] = data['label'][step_begin:step_end, :]
                result['velocity'] = data['velocity'][step_begin:step_end, :]
            else:
                result['audio'] = data['audio'][:]
                result['label'] = data['label'][:]
                result['velocity'] = data['velocity'][:]

            result['audio'] = torch.tensor(result['audio'])
            result['label'] = torch.tensor(result['label'])
            result['velocity'] = torch.tensor(result['velocity'])


            result['audio'] = result['audio'].float().div_(32768.0)
            result['onset'] = (result['label'] == 3).float()
            result['offset'] = (result['label'] == 1).float()
            result['frame'] = (result['label'] > 1).float()
            result['velocity'] = result['velocity'].float().div_(128.0)

            # Down sampling
            # result['onset'] = max_pooling(result['onset'], [2, 1])
            # result['offset'] = max_pooling(result['offset'], [2, 1])
            # result['frame'] = max_pooling(result['frame'], [2, 1])
            # result['velocity'] = max_pooling(result['velocity'], [2, 1])

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        
        # h5_path = audio_path.replace('.flac', '.h5').replace('.wav', '.h5').replace('.mp3', '.h5')

        root,basename = os.path.split(audio_path)
        name, ext = os.path.splitext(basename)
        h5_dir = os.path.join(root, '..', f'h5_hop{HOP_LENGTH}')
        os.makedirs(h5_dir, exist_ok=True)
        h5_path = os.path.join(h5_dir, name + '.h5')

        if os.path.exists(h5_path):
            return h5_path


        with h5py.File(h5_path, mode='w') as h5:                
            audio, sr = soundfile.read(audio_path, dtype='int16')
            # audio, sr = librosa.load(audio_path, sr = SAMPLE_RATE, mono=True)
            assert sr == SAMPLE_RATE

            audio = torch.ShortTensor(audio)
            audio_length = len(audio)

            n_keys = MAX_MIDI - MIN_MIDI + 1
            n_steps = (audio_length - 1) // HOP_LENGTH + 1

            label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
            velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)


            if(len(tsv_path) > 0):
                tsv_path = tsv_path
                midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

                for onset, offset, note, vel in midi:
                    left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                    onset_right = min(n_steps, left + HOPS_IN_ONSET)
                    frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                    frame_right = min(n_steps, frame_right)
                    offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                    f = int(note) - MIN_MIDI
                    label[left:onset_right, f] = 3
                    label[onset_right:frame_right, f] = 2
                    label[frame_right:offset_right, f] = 1
                    velocity[left:frame_right, f] = vel

            # data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
            h5['path'] = audio_path
            h5['audio'] = audio.numpy()
            h5['label'] = label.numpy()
            h5['velocity'] = velocity.numpy()
            # torch.save(data, saved_data_path)
        # return data
        return h5_path

class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/maestro-v3.0.0', groups=None, sequence_length=None, seed=42):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            # metadata = json.load(open(os.path.join(self.path, 'maestro-v3.0.0.json')))
            meta_df = pd.read_csv(os.path.join(self.path, 'maestro-v3.0.0.csv'))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for ind,row in meta_df.iterrows() if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files if os.path.exists(audio)]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        # tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        tsvs = [f.replace('/flac/', '/midi/').replace('.flac', '.tsv') for f in flacs]


        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))




