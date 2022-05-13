import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
import time

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
import mir_eval
# from pyrsistent import v
from scipy.stats import hmean
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import h5py
import pandas as pd

from torch.utils.data import Subset 

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *

eps = sys.float_info.epsilon

def get_pred_lst(dataset, model, device, save_path, clip_len=10240):
    # outputs:
    #   pred = [ {"onset":ndarray, "offset":..., "frame":..., "velocity":...}, {}, {}, ... ]
    #   losses = [{"loss/onset": scalar}, {}, ..., {}]
    # 
    pred_lst = []
    loss_lst = []
    assert len(dataset) == len(dataset.data)
    for i in tqdm(range(len(dataset)), desc="getting pred"):

        # label['path'] = str(label['path'])
        label_path = dataset.data[i]


        os.makedirs(save_path, exist_ok=True)
        pred_path = os.path.join(save_path, os.path.basename(label_path)) # + '.pred.h5'
        pred_path = pred_path.replace("flac", "flac'")
        pred_path = pred_path.replace(".h5", ".flac'.pred.h5")
        # print("pred_path", pred_path)
        # load previous pred
        if(os.path.exists(pred_path)):
            continue
            pred = {'onset':None, 'offset':None, 'frame':None, 'velocity': None}
            losses = {'loss/onset': None, 'loss/offset': None, 'loss/frame': None, 'loss/velocity':None}
            with h5py.File(pred_path, 'r') as h5:
                for key in pred:
                    pred[key] = h5[key][:] / 255.0
                for key in losses:
                    if(key in h5):
                        # losses[key] = torch.tensor(h5[key][()]).to(device)
                        losses[key] = h5[key][()]
                    else:
                        losses[key] = 0
        # get new pred
        else:
            label = dataset[i]
            n_step =  label['onset'].shape[-2]
            
            label['audio'] = label['audio'].to(device)
            label['onset'] = label['onset'].to(device).float()
            label['offset'] = label['offset'].to(device).float()
            label['frame'] = label['frame'].to(device).float()
            label['velocity'] = label['velocity'].to(device)

            # 
            if(len(label['audio'].size()) > 1 or n_step <= clip_len):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    pred, losses = model.run_on_batch(label)
            # clip audio to fixed length to prevent out of memory.
            else: # when test on long audio
                print('n_step > clip_len %d '%clip_len, label['audio'].shape, label['onset'].shape)
                clip_list = [clip_len] * (n_step // clip_len)
                res = n_step % clip_len
                if(n_step > clip_len and res != 0):
                    clip_list[-1] -= (clip_len - res)//2
                    clip_list += [res + (clip_len - res)//2]

                print('clip list:', clip_list)

                begin = 0
                pred = {}
                losses = {}
                for clip in clip_list:
                    end = begin + clip
                    label_i = {}
                    label_i['audio'] = label['audio'][HOP_LENGTH*begin:HOP_LENGTH*end]
                    label_i['onset'] = label['onset'][begin:end]
                    label_i['offset'] = label['offset'][begin:end]
                    label_i['frame'] = label['frame'][begin:end]
                    label_i['velocity'] = label['velocity'][begin:end]
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        pred_i, losses_i = model.run_on_batch(label_i)

                    for key, item in pred_i.items():
                        if(key in pred):
                            pred[key] = torch.cat([pred[key], item], dim=0)
                        else:
                            pred[key] = item

                    for key, loss in losses_i.items():
                        if(key in losses):
                            losses[key] += loss * clip / n_step
                        else:
                            losses[key] = loss * clip / n_step
                    begin += clip
            # save pred
            if(not save_path is None):
                with h5py.File(pred_path, 'w') as h5:
                    for key, item in pred.items():
                        
                        pred[key] = torch.squeeze(item,dim=0).cpu().numpy()
                        data = (item*255).to(torch.uint8).cpu().numpy()
                        h5.create_dataset(key, data.shape, np.uint8, data, compression=9)
                        # h5[key] = data
                    for key, item in losses.items():
                        h5[key] = losses[key] = item.cpu().item()

        # loss_lst.append(losses)

        # for key, value in pred.items():
        #     value.squeeze_(0).relu_()

        # pred_lst.append(pred)

    return  pred_lst, loss_lst



def cal_score(sample_data):
    dataset = torch.load('dataset.tmp.pth')

    device = sample_data['device']
    
    #
    metric_dict = {}

    sample_id = sample_data['sample_id']
    # label = sample_data['label']
    label = dataset[sample_id]

    # load pred in h5
    label['path'] = str(label['path'])

    save_path = sample_data['save_path']
    pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.h5')
    pred_path = pred_path.replace("flac", "flac'")
    # load previous pred

    pred = {'onset':None, 'offset':None, 'frame':None, 'velocity': None}
    losses = {'loss/onset': None, 'loss/offset': None, 'loss/frame': None, 'loss/velocity':None}
    with h5py.File(pred_path, 'r') as h5:
        for key in pred:
            pred[key] = h5[key][:] / 255.0
        for key in losses:
            if(key in h5):
                # losses[key] = torch.tensor(h5[key][()]).to(device)
                losses[key] = h5[key][()]
            else:
                losses[key] = 0


    # pred = sample_data['pred']
    
    # losses = sample_data['losses']
    
    onset_threshold = sample_data['onset_threshold']
    frame_threshold = sample_data['frame_threshold']

    # load saved metric json
    metric_json_path = os.path.join(save_path, "%03d.json"%sample_id)
    if(os.path.exists(metric_json_path)):
        with open(metric_json_path, 'r') as f:
            metric_dict = eval(f.read())
            
        return metric_dict


    # label['audio'] = label['audio'].to(device) # use [0] to unbach
    label['onset'] = label['onset'].to(device)
    label['offset'] = label['offset'].to(device)
    label['frame'] = label['frame'].to(device)
    label['velocity'] = label['velocity'].to(device)

    pred['onset'] = torch.tensor(pred['onset']).to(device)
    pred['offset'] = torch.tensor(pred['offset']).to(device)
    pred['frame'] = torch.tensor(pred['frame']).to(device)
    pred['velocity'] = torch.tensor(pred['velocity']).to(device)

    

    # dynamic threshold of onset
    # pred['velocity'] = pred['velocity'] * (pred['onset'] > 0.2).float()
    # pred['onset'] = (pred['onset'] / torch.clip(pred['velocity'], 0.8, 1.0))

    label['path'] = str(label['path'])

    for key, loss in losses.items():
        metric_dict[key] = loss

    # for key, value in pred.items():
    #     value.squeeze_(0).relu_()

    # pitch, interval, velocity
    p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
    p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], onset_threshold, frame_threshold)


    # time, frequency
    t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
    t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    # print('max_interval', max(i_ref[:,1]-i_ref[:, 0]))
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])


    ############################################
    # # find what ref notes are not matched.
    # matched_list = mir_eval.transcription.match_note_onsets(i_ref, i_est)
    # matched_ref_list = [m[0] for m in matched_list]
    # with open('not_matched_note.txt', 'a') as f:
    #     # f.write('total notes:%d\n'%(len(p_est)))
    #     for i in range(len(p_ref)):
    #         if not i in matched_ref_list:
    #             f.write('%d,%.3f,%.3f\n'%(note_ref[i], i_ref[i][1]-i_ref[i][0], v_ref[i]))
    #########################################3

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metric_dict['metric/note/precision'] = p
    metric_dict['metric/note/recall'] = r
    metric_dict['metric/note/f1'] = f
    metric_dict['metric/note/overlap'] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metric_dict['metric/note-with-offsets/precision'] = p
    metric_dict['metric/note-with-offsets/recall'] = r
    metric_dict['metric/note-with-offsets/f1'] = f
    metric_dict['metric/note-with-offsets/overlap'] = o

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                offset_ratio=None, velocity_tolerance=0.1)
    metric_dict['metric/note-with-velocity/precision'] = p
    metric_dict['metric/note-with-velocity/recall'] = r
    metric_dict['metric/note-with-velocity/f1'] = f
    metric_dict['metric/note-with-velocity/overlap'] = o

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
    metric_dict['metric/note-with-offsets-and-velocity/precision'] = p
    metric_dict['metric/note-with-offsets-and-velocity/recall'] = r
    metric_dict['metric/note-with-offsets-and-velocity/f1'] = f
    metric_dict['metric/note-with-offsets-and-velocity/overlap'] = o

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metric_dict['metric/frame/f1'] = hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps

    for key, loss in frame_metrics.items():
        metric_dict['metric/frame/' + key.lower().replace(' ', '_')] =loss


    if metric_dict is not None:
        # os.makedirs(save_path, exist_ok=True)
        label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
        save_pianoroll(label_path, label['onset'], label['frame'], onset_threshold, frame_threshold, zoom=1)
        pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
        save_pianoroll(pred_path, pred['onset'], pred['frame'], onset_threshold, frame_threshold, zoom=1)
        midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)

        frame_overlap_path = pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.overlap.note.png')
        utils.save_pianoroll_overlap(frame_overlap_path, label['frame'], pred['frame'], frame_threshold, zoom=1)

        onset_overlap_path = pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.onset_overlap.png')
        utils.save_pianoroll_overlap(onset_overlap_path, label['onset'], pred['onset'], onset_threshold, zoom=1)

        pred_onset_path = pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.onset.png')
        plt.imsave(pred_onset_path, pred['onset'].cpu().numpy())

        plt.imsave(os.path.join(save_path, os.path.basename(label['path'])+".pred.frame.png"), torch.swapdims(pred['frame'], 0,1))
        
        frame_cat = torch.cat([label['frame'], pred['frame']], dim=1)
        frame_cat = torch.swapdims(frame_cat, 0, 1)
        plt.imsave(os.path.join(save_path, os.path.basename(label['path'])+".compare.frame.png"), frame_cat)

    print("=================================")
    print("sample done:", sample_id)
    duration = time.time() - sample_data['begin_time']
    print("duration: ", int(duration//(60)), "min ", int(duration % 60), "s")

    with open(os.path.join(save_path, "%03d.json"%sample_id), "w") as f:
        new_dict = {}
        new_dict.update(metric_dict)
        new_dict['path'] = os.path.split(label['path'])[1]
        f.write(str(new_dict).replace("'", '"'))
    return metric_dict


def evaluate(dataset, model, device, onset_threshold=0.5, frame_threshold=0.5, save_path=None, save_metrics_only=False, clip_len=10240, pool_num = 5):
    metrics = defaultdict(list)

    print('getting pred list ...')
    pred_lst, loss_lst = get_pred_lst(dataset, model, device, save_path, clip_len)
    print('pred_lst.len:', len(pred_lst))

    print('evaluating pred list ...')


    sample_data_list = []


    

    for i in range(len(dataset)):
        

        # sample_metric_json_path = os.path.join(save_path, "%03d.json"%i)
        # if(os.path.exists(sample_metric_json_path)):
        #     with open(sample_metric_json_path, 'r') as f:
        #         metric_dict = eval(f.read())
                
        #     continue
        # label = data[i]

        # label['audio'] = label['audio'].to(device) # use [0] to unbach
        # label['onset'] = label['onset'].cpu()
        # label['offset'] = label['offset'].cpu()
        # label['frame'] = label['frame'].cpu()
        # label['velocity'] = label['velocity'].cpu()

        # label['path'] = str(label['path'])

        # pred = pred_lst[i]
        # losses = loss_lst[i]

        sample_data = {}
        # del label['audio']
        # sample_data['label'] = label
        # sample_data['pred'] = pred
        sample_data['device'] = 'cpu'
        # sample_data['losses'] = losses
        sample_data['save_path'] = save_path
        sample_data['onset_threshold'] = onset_threshold
        sample_data['frame_threshold'] = frame_threshold
        sample_data['sample_id'] = i
        sample_data['begin_time'] = time.time()

        sample_data_list.append(sample_data)

    torch.save(dataset, 'dataset.tmp.pth')

    
    print(f'use Pool(f{pool_num})')
    with Pool(pool_num) as pool:
        metric_list = list(tqdm(pool.imap(cal_score, sample_data_list), total=len(sample_data_list)))
        
    for metric_dict in tqdm(metric_list, desc="listing metrics"):
        for key, value in metric_dict.items():
            metrics[key].append(value)


    # print('use single thread')
    # metric_list = []
    # for s in sample_data_list:
    #     metric_list.append(cal_score(s))
    # for metric_dict in tqdm(metric_list, desc="listing metrics"):
    #     for key, value in metric_dict.items():
    #         metrics[key].append(value)


    return metrics


def evaluate_file(model_file, dataset_name, dataset_group, sequence_length, save_path,
                  onset_threshold, frame_threshold, device, clip_len=10240, pool_num = 5):


    dataset_class = getattr(dataset_module, dataset_name)
    kwargs = {'sequence_length': sequence_length} # , 'device': device
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    if(save_path == None):
        group_str = dataset_group if dataset_group is not None else 'default'
        if(dataset_name == "MAESTRO"):
            dataset_name = os.path.basename(dataset.path)
        save_path = os.path.join(model_file[:-3] + "_evaluate", dataset_name, group_str)

    # offset = 0
    # dataset = Subset(dataset, list(range(offset,offset+10)))

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    metrics = evaluate(dataset, model, device, onset_threshold, frame_threshold, save_path, save_metrics_only=False, clip_len=clip_len, pool_num= pool_num)

    

    res = '\n' + model_file +   '\n' + datetime.now().strftime('%y%m%d-%H%M%S') + '\n\nMetrics:\n'
    res += 'evaluate dataset and group:' + str(dataset) + ', ' + str(dataset_group) + '\n'
    res += 'audio piece num: %d\n'%(len(dataset))
    res += 'sequence_len: %s, clip_len: %d\n'%(str(sequence_length), clip_len)
    res += 'onset and frame threshold: %f, %f'%(onset_threshold, frame_threshold) + '\n'


    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            res += '\n' + f'{category:>32} {name:25}: {np.mean(values)*100:.2f} ± {np.std(values)*100:.2f}'
    print(res)

    if(save_path != None):
        result_path = os.path.join(save_path, 'metrics_result.txt')
        with open(result_path, 'a') as f:
            f.write(res)

        # save metrics to csv
        column_dict = {}
        for key, values in metrics.items():
            # metric/note-with-offsets-and-velocity/f1
            if(key.find('loss') >= 0):
                continue
            new_key = key
            replace = {'metric/':'', '-with':'', '-and':'', 'onsets': 'on', 'offsets':'off', 'velocity':'vel'}
            for k,v in replace.items():
                new_key = new_key.replace(k, v)
            column_dict[new_key] = values
        column_dict['path'] = [os.path.split(str(data['path']))[-1] for data in dataset]
        df = pd.DataFrame.from_dict(column_dict)
        csv_path = os.path.join(save_path, 'metrics_result.csv')
        print('save to :', csv_path)
        df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset_name', nargs='?', default='MAPS')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.4, type=float)
    parser.add_argument('--frame-threshold', default=0.4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--device', default='cpu')
    parser.add_argument('--pool_num', default=5, type=int)

    with torch.no_grad():
        evaluate_file(**vars(parser.parse_args()))
