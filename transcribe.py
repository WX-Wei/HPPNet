import argparse
import os
import sys

import numpy as np
import soundfile
from mir_eval.util import midi_to_hz

from hppnet import *
import librosa


def load_and_process_audio(flac_path, sequence_length, device):

    # random = np.random.RandomState(seed=42)

    # audio, sr = soundfile.read(flac_path, dtype='int16')
    # assert sr == SAMPLE_RATE

    # audio = torch.ShortTensor(audio)

    # if sequence_length is not None:
    #     audio_length = len(audio)
    #     step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
    #     n_steps = sequence_length // HOP_LENGTH

    #     begin = step_begin * HOP_LENGTH
    #     end = begin + sequence_length

    #     audio = audio[begin:end].to(device)
    # else:
    #     audio = audio.to(device)

    # audio = audio.float().div_(32768.0)
    
    audio, sr = librosa.load(flac_path, sr=16000)
    

    return torch.tensor(audio)


def transcribe(model, audio):

    # mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    # onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)
    
    # predictions = {
    #         'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
    #         'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
    #         'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
    #         'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
    #     }
    
    # return predictions

    res = model(audio, torch.zeros([100, 352]), inference_mode=True)
    
    res['onset'] = torch.squeeze( res['onset'], dim=0 )
    res['offset'] = torch.squeeze( res['offset'], dim=0 )
    res['frame'] = torch.squeeze( res['frame'], dim=0 )
    res['velocity'] = torch.squeeze( res['velocity'], dim=0 )
    
    return res;


def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device):

    model = torch.load(model_file, map_location=device).eval()
    model.config['device'] = device
    summary(model)

    midi_paths = []
    for flac_path in flac_paths:
        print(f'Processing {flac_path}...', file=sys.stderr)
        audio = load_and_process_audio(flac_path, sequence_length, device)
        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)
        pred_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])
        midi_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)
        midi_paths.append(midi_path)
    return midi_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('model_file', type=str)
    
    parser.add_argument('flac_paths', type=str, nargs='+')
    parser.add_argument('--model_file', type=str, default="checkpoints/model-220519-005612-tiny-498000.pt")
    parser.add_argument('--save-path', type=str, default='test')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.4, type=float)
    parser.add_argument('--frame-threshold', default=0.4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
