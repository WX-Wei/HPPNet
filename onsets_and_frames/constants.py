import torch


SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

N_HAR = 8
BINS_PER_SEMITONE = 4
MODEL_COMPLEXITY = 48 # 12 # model_complexity

DEFAULT_DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'

# SUB_NETS = ['onset', 'offset', 'frame', 'velocity']
SUB_NETS = ['onset']
SUB_NETS = ['onset', 'offset', 'frame']
