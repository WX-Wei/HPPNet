import os
from datetime import datetime
from re import sub, subn
import copy

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch
import torch.utils.data
import torch.cuda
import torchaudio
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchsummary

from tqdm import tqdm
import matplotlib.pyplot as plt
from random_words import RandomWords

from evaluate import evaluate
from onsets_and_frames import *

rw = RandomWords()
random_word_str = rw.random_word()
time_str = datetime.now().strftime('%y%m%d-%H%M%S') + '_' + random_word_str
ex = Experiment('train_transcriber')
ex.time_str = time_str

mongo_ob = MongoObserver.create(url='10.177.55.66:7000', db_name='piano_transcription') #harmonic_net_mono
ex.observers.append(mongo_ob)

ex.tags = []





@ex.config
def config():
    logdir = 'runs/transcriber-' + time_str
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 200*1000
    resume_iteration = None
    checkpoint_interval = 2000
    train_on = 'MAESTRO'

    batch_size = 2
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 400

    test_interval= None

    
    ex.observers.append(FileStorageObserver.create(logdir))


    training_size = 1.0 # [1.0, 0.3, 0.1] preportion used for training in training set.

    notes = ""


@ex.config
def model_config():
    SUB_NETS = ['onset', 'frame', 'velocity']
    model_name = "HPP" # modeling harmonic structure and pitch invariance in piano transcription
    head_type = 'FB-LSTM' # 'LSTM', 'Conv'
    trunk_type = 'HD-Conv' # 'SD-Conv', 'Conv'

    fixed_dilation = 24

    model_size = 128

@ex.config
def train_with_test():
    # validation_interval = 50
    test_interval = 50000

    test_onset_threshold = 0.4
    test_frame_threshold = 0.3

@ex.named_config
def train_without_test():
    test_interval = None

@ex.named_config
# baseline 'onsets&frames'
def train_baseline():
    SUB_NETS = ['all']
    model_name='onsets&frames'
    model_size = 48 * 16
    iterations = 500*1000
    checkpoint_interval = 20000

    batch_size=4



    # batch_size = 2



#####################################
# ablation study

# @ex.command
# def empty_cmd(device = 'cpu'):
#     print('empty command.')
#     device='cpu'


ex.main_locals = locals()


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
           learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, 
          test_interval, test_onset_threshold, test_frame_threshold, 
          training_size,
          model_name):
    print_config(ex.current_run)

    config = ex.current_run.config

    SUB_NETS = config['SUB_NETS']


    # add source files to ex
    src_file_set = set()
    src_file_dir = os.path.join(ex.observers[1].dir, 'src')
    utils.save_src_files(ex.main_locals, src_file_dir, query_str='onsets-and-frames', src_path_set=src_file_set)
    for src_path in src_file_set:
        ex.add_source_file(src_path)

    utils.copy_dir('./', src_file_dir)
    utils.copy_dir('./onsets_and_frames', os.path.join(src_file_dir, 'onsets_and_frames'))


    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    ex.basedir = ex.current_run.observers[1].basedir

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
        
        # test
        test_dataset = MAESTRO(groups=['test'])
        
        # groups = test_dataset.groups
        # test_dataset = torch.utils.data.Subset(test_dataset, list(range(3)))
        # test_dataset.groups = groups
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)
        test_dataset = MAPS(groups=[['ENSTDkAm', 'ENSTDkCl']])

    train_idx = [int(x/training_size) for x in range(int(len(dataset)*training_size))]
    ex.info['training_files'] = dataset.files('train')
    ex.info['training_idx'] = train_idx
    dataset = torch.utils.data.Subset(dataset, train_idx)
    ex.info['train_num'] = len(dataset) 

    ex.info['validation_set_files'] = validation_dataset.files('validation')
    ex.info['test_set_files'] = test_dataset.files('test')

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    # validation_dataset = DataLoader(validation_dataset, num_workers=4)


    optimizers = {}
    if resume_iteration is None:
        if(model_name=='onsets&frames'):
            model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1).to(device)
            model.sub_nets = {}
            model.sub_nets['all'] = torch.nn.ModuleList([x for x in  model.modules()])
        else:
            model = HARPIST(N_MELS, MAX_MIDI - MIN_MIDI + 1, config).to(device)

        # optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        for subnet in SUB_NETS:
            optimizers[subnet] = torch.optim.Adam(model.sub_nets[subnet].parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        # optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
        for subnet in SUB_NETS:
            optimizers[subnet] = torch.optim.Adam(model.sub_nets[subnet].parameters(), learning_rate)
            optimizers[subnet].load_state_dict(torch.load(os.path.join(logdir, f'last-optimizer-state-{subnet}.pt')))
            
    # summary
    # torchsummary.summary(model, input_size=(1, 16000*4, ), batch_size=1, device='cpu')
    # writer.add_graph(model, torch.zeros([2, 16000*20]))
    # summary(model)
    summary_path = ex.basedir + '/model_summary.txt'
    summary(model, summary_path)
    ex.add_artifact(summary_path)

    # scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    schedulers = {}
    for subnet in SUB_NETS:
        schedulers[subnet] = StepLR(optimizers[subnet], step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    loop.set_description(config['model_name'] + '_' + random_word_str)
    tqdm_dict = {}
    for i, batch in zip(loop, cycle(loader)):

        batch['audio'] = batch['audio'].to(device)
        batch['onset'] = batch['onset'].to(device)
        batch['offset'] = batch['offset'].to(device)
        batch['frame'] = batch['frame'].to(device)
        batch['velocity'] = batch['velocity'].to(device)



        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())

        if(model_name=='onsets&frames'):
            losses[f'loss/all'] = loss
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

        for subnet in SUB_NETS:
            loss_subnet = losses[f'loss/{subnet}']
            optimizers[subnet].zero_grad()
            loss_subnet.backward()
            optimizers[subnet].step()
            schedulers[subnet].step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)
            ex.log_scalar(key, value.item(),i)

        if(i %10 == 0):
            tqdm_dict['train/loss'] = loss.cpu().detach().numpy()
            loop.set_postfix(tqdm_dict)

        if(i in [100, 1000, 2000, 4000, 8000] or i % 10000 == 0 ):
            frame_img_pred = torch.swapdims(predictions['frame'], 1, 2)
            frame_img_pred = torch.unsqueeze(frame_img_pred, dim=1)
            # => [F x T]
            frame_img_pred = torchvision.utils.make_grid(frame_img_pred, pad_value=0.5)
            # writer.add_image('train/step_%d_pred'%i, frame_img_pred)

            frame_img_ref = torch.swapdims(batch['frame'], 1, 2)
            frame_img_ref = torch.unsqueeze(frame_img_ref, dim=1)
            frame_img_ref = torchvision.utils.make_grid(frame_img_ref, pad_value=0.5)
            # writer.add_image('train/step_%d_ref'%i, frame_img_ref)

            frame_img = torch.cat([frame_img_ref[0], frame_img_pred[0]], dim=0)
            dir_path = os.path.join(logdir, 'piano_roll')
            os.makedirs(dir_path, exist_ok=True)
            plt.imsave(dir_path + '/train_step_%d.png'%(i), frame_img.detach().cpu().numpy())

        ##################################
        # Validate
        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate(validation_dataset, model, device)
                for key, value in val_metrics.items():
                    mean_val = np.mean(value)
                    writer.add_scalar('validation/' + key.replace(' ', '_'), mean_val, global_step=i)
                    ex.log_scalar('validation/' + key.replace(' ', '_'), mean_val, i)
                tqdm_dict['on_loss'] = '%.4f'%np.mean(val_metrics['loss/onset'])
                tqdm_dict['f_f1'] = '%.1f'%np.mean(val_metrics['metric/frame/f1']) * 100
                tqdm_dict['n_f1'] = '%.1f'%np.mean(val_metrics['metric/note/f1']) * 100
                loop.set_postfix(tqdm_dict)
            model.train()

        ##################################
        # Test
        if not test_interval is None:
            if i % test_interval == 0:
                model.eval()
                clip_len = 10240
                test_result = {}
                test_result['step'] = i
                test_result['time'] = datetime.now().strftime('%y%m%d-%H%M%S')
                test_result['dataset'] = str(test_dataset)
                test_result['dataset_group'] = test_dataset.groups
                test_result['dataset_len'] = len(test_dataset)
                test_result['clip_len'] = clip_len
                test_result['onset_threshold'] = test_onset_threshold
                test_result['frame_threshold'] = test_frame_threshold
                with torch.no_grad():
                    eval_result =  evaluate(test_dataset, model, device,
                        onset_threshold=test_onset_threshold, frame_threshold=test_frame_threshold,
                        clip_len = clip_len,
                        save_path=config['logdir'] + f'/model-{i}-test'
                    )
                    for key, values in eval_result.items():
                        mean_val = np.mean(values)
                        # std_val = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                        label = 'test/' + key.replace(' ', '_')
                        writer.add_scalar(label, mean_val, global_step=i)
                        ex.log_scalar(label, mean_val, i)
                        test_result[label] = "%.2f"%(mean_val*100)
                ex.info[f'test_step_{i}'] = test_result
                model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))

            # torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            for subnet in SUB_NETS:
                torch.save(optimizers[subnet].state_dict(), os.path.join(logdir, f'last-optimizer-state-{subnet}.pt'))
