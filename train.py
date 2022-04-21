import os
from datetime import datetime
from re import sub, subn

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchaudio
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from evaluate import evaluate
from onsets_and_frames import *


ex = Experiment('train_transcriber')

mongo_ob = MongoObserver.create(url='10.177.55.66:7000', db_name='piano_transcription') #harmonic_net_mono
ex.observers.append(mongo_ob)

ex.tags = []


@ex.config
def config():
    # logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + "_MRCD-Conv_BiLSTM[freq->LMH,CQT,full_maestro,on_off_vel_use_baseline]"
    # logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + "_MRD-Conv_BILSTM[no_combined_res_connect_freq->LMH,CQT,kernel_size=5,layernorm,hop_len20ms,onset2]"
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + "_MRD-Conv_BILSTM[onset_only,soft_label]"
    # logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + "_MRD-Conv_BILSTM[onset_only,log_specgram2]"
    # logdir = 'data/runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + "_MRD-Conv_BILSTM[onset_only,time_pooling4]"
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + "_HD-Conv[onset_frame_vel_sep_0413,weighted_loss_2,LSTM,maestro-v3]"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500*1000
    resume_iteration = None
    checkpoint_interval = 2000
    train_on = 'MAESTRO'

    batch_size = 2
    sequence_length = 327680
    model_complexity = constants.MODEL_COMPLEXITY

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

    ex.observers.append(FileStorageObserver.create(logdir))

    training_size = 1.0 # [1.0, 0.3, 0.1]



@ex.config
def model_config():
    model_name = "HPP" # modeling harmonic structure and pitch invariance in piano transcription
    head_type = 'FB-LSTM' # 'LSTM', 'None'
    trunk_type = 'HD-Conv' # 'SD-Conv', 'None'

    model_size = 128


#####################################
# ablation study


@ex.named_config
def baseline_onsets_and_frames():
    model_name = 'onsets&frames'

    


ex.main_locals = locals()


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    print_config(ex.current_run)


    # add source files to ex
    src_file_set = set()
    src_file_dir = os.path.join(ex.observers[0].dir, 'src')
    utils.save_src_files(ex.main_locals, src_file_dir, query_str='onsets-and-frames', src_path_set=src_file_set)
    for src_path in src_file_set:
        ex.add_source_file(src_path)

    utils.copy_dir('./', src_file_dir)
    utils.copy_dir('./onsets_and_frames', os.path.join(src_file_dir, 'onsets_and_frames'))


    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)



    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

    # validation_dataset = DataLoader(validation_dataset, num_workers=4)


    optimizers = {}
    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, ex.current_run.config).to(device)
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
            
    summary(model)
    # scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    schedulers = {}
    for subnet in SUB_NETS:
        schedulers[subnet] = StepLR(optimizers[subnet], step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):

        batch['audio'] = batch['audio'].to(device)
        batch['onset'] = batch['onset'].to(device)
        batch['offset'] = batch['offset'].to(device)
        batch['frame'] = batch['frame'].to(device)
        batch['velocity'] = batch['velocity'].to(device)



        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
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

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model, device).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
                    ex.log_scalar('validation/' + key.replace(' ', '_'), np.mean(value), i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))

            # torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            for subnet in SUB_NETS:
                torch.save(optimizers[subnet].state_dict(), os.path.join(logdir, f'last-optimizer-state-{subnet}.pt'))
