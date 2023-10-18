# PyTorch Implementation of HPPNet Piano Transcription Model


This is a [PyTorch](https://pytorch.org/) implementation of [HPPNet](https://arxiv.org/abs/2208.14339) model, using the [Maestro dataset v3](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing.



## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 

### Downloading Dataset

To download the Maestro dataset, first make sure that you have `ffmpeg` executable and run `prepare_maestro.sh` script:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full Maestro dataset from Google's server and automatically unzip and encode them as FLAC files in order to save storage. However, you'll still need about 200 GB of space for intermediate storage.

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame metrics:

```bash

python evaluate.py runs/transcriber/model-600000.pt MAPS test
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py runs/model/model-100000.pt --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py runs/transcriber/model-600000.pt MAESTRO test
```
## Acknowledgements

This project is based on the PyTorch implementation of Onsets and Frames model => https://github.com/jongwook/onsets-and-frames


## Citation

```
@inproceedings{Wei2022HPPNet,
  author       = {Weixing Wei and
                  Peilin Li and
                  Yi Yu and
                  Wei Li},
  title        = {HPPNet: Modeling the Harmonic Structure and Pitch Invariance in Piano
                  Transcription},
  booktitle    = {Proceedings of the 23rd International Society for Music Information
                  Retrieval Conference, {ISMIR} 2022, Bengaluru, India, December 4-8,
                  2022},
  pages        = {709--716},
  year         = {2022},
  url          = {https://archives.ismir.net/ismir2022/paper/000085.pdf},
}
```
