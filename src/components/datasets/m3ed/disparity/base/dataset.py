import os
from PIL import Image
from skimage.morphology import square, dilation
from pathlib import Path
import numpy as np
import yaml

import torch.utils.data
from .utils import numpy_sample_hints


class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'raw_mae': 'raw_event_mae_filtered.txt',
        'event': 'event',
        'event_raw': 'event_raw',
        'disp_factor': 'disp_factor.yaml',
        'skip_frames': 'skip_frames.yaml'
    }
    _DOMAIN = ['event', 'event_raw']
    _FRAME_SKIP_START = 50
    _FRAME_SKIP_END = 50
    NO_VALUE = 0.0

    def __init__(self, root, guideperc, randomperc, raw_frameratio, seq_size, raw_offset_us, raw_mae_threshold, max_disparity_threshold, rnd_sampling, skip_frames, dilation = 1):
        self.root = root
        self.guideperc = guideperc
        self.randomperc = randomperc
        self.raw_frameratio = raw_frameratio
        self.seq_size = seq_size
        self.raw_offset_us = raw_offset_us
        self.raw_frameratio_counter = 0
        self.rnd_sampling = rnd_sampling
        self.skip_frames = skip_frames
        self.dilation = dilation
        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))
        self.sequence_name = os.path.basename(os.path.abspath(os.path.join(root,"..")))

        try:
            raw_maes = np.loadtxt(os.path.join(root, self._PATH_DICT['raw_mae']))
        except:
            raw_maes = None

        self.raw_maes = raw_maes

        if os.path.exists(os.path.join(root, self._PATH_DICT['disp_factor'])):
            with open(os.path.join(root, self._PATH_DICT['disp_factor']), 'r') as f:
                self.disp_factor = yaml.safe_load(f)
        else:
            self.disp_factor = None

        if skip_frames:
            if os.path.exists(os.path.join(root, self._PATH_DICT['skip_frames'])):
                with open(os.path.join(root, self._PATH_DICT['skip_frames']), 'r') as f:
                    tmp_dict = yaml.safe_load(f)
                    self.frame_skip_start = tmp_dict['frame_skip_start']
                    self.frame_skip_stop = tmp_dict['frame_skip_stop']
            else:
                self.frame_skip_start = self._FRAME_SKIP_START
                self.frame_skip_stop = self._FRAME_SKIP_END
        else:
            self.frame_skip_start = 0
            self.frame_skip_stop = 0

        self.disparity_path_list = {}
        self.timestamp_to_disparity_path = {}
        
        for domain in self._DOMAIN:
            if domain == 'event_raw':
                if self.rnd_sampling:
                    self.disparity_path_list[domain] = get_path_list(os.path.join(root, f"{self._PATH_DICT['event']}_{raw_offset_us}"))
                else:
                    self.disparity_path_list[domain] = get_path_list(os.path.join(root, f"{self._PATH_DICT['event_raw']}_{raw_offset_us}"))
            else:
                self.disparity_path_list[domain] = get_path_list(os.path.join(root, self._PATH_DICT[domain]))
                
            self.timestamp_to_disparity_path[domain] = {timestamp: filepath for timestamp, filepath in
                                                        zip(self.timestamps, self.disparity_path_list[domain])}
        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_disparity_path['event'][timestamp]))[0])
            for timestamp in self.timestamp_to_disparity_path['event'].keys()
        }

        #Filter accordingly to arg and skip parameters 
        tmp_timestamps = []
        for k in range(len(self.raw_maes)):
            if self.frame_skip_start <= k <= len(self.timestamps)-self.frame_skip_stop:
                if (( raw_mae_threshold is None or raw_mae_threshold <= 0 or self.raw_maes[k] <= raw_mae_threshold)
                    and (max_disparity_threshold is None or max_disparity_threshold <= 0 or np.max(load_disparity(self.timestamp_to_disparity_path['event'][self.timestamps[k]],self.disp_factor)) <= max_disparity_threshold)):
                    tmp_timestamps.append(self.timestamps[k])
                
        self.timestamps = np.array(tmp_timestamps)

        if self.seq_size > 0:
            mysize = min(len(self.timestamps), self.seq_size)
            self.timestamps = self.timestamps[:mysize]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        disp = load_disparity(self.timestamp_to_disparity_path['event'][timestamp],self.disp_factor)

        if self.raw_frameratio_counter % self.raw_frameratio == 0:
            if self.rnd_sampling:
                if self.randomperc:
                    guideperc = np.random.choice(range(0,6))*0.03
                else: 
                    guideperc = self.guideperc

                #Load event_{offset} only if necessary
                if self.raw_offset_us > 0:
                    raw_disp = load_disparity(self.timestamp_to_disparity_path['event_raw'][timestamp],self.disp_factor)
                else:
                    raw_disp = disp

                raw_disp = numpy_sample_hints(raw_disp, raw_disp>0, guideperc)
            else:
                raw_disp = load_disparity(self.timestamp_to_disparity_path['event_raw'][timestamp],self.disp_factor)
        else:
            raw_disp = np.zeros_like(disp)

        if self.dilation > 1:
            raw_disp = dilation(raw_disp, square(self.dilation))        

        self.raw_frameratio_counter += 1
        
        return disp, raw_disp

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch


def load_timestamp(root):
    return np.loadtxt(root, dtype='int64')


def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]


def load_disparity(root, factor_dict = None):
    if factor_dict is not None:
        dir = os.path.dirname(root).split("/")[-1]
        filename = os.path.basename(root)
        disparity = np.array(Image.open(root)).astype(np.float32) / factor_dict[dir][filename]
    else:
        disparity = np.array(Image.open(root)).astype(np.float32) / 256.0
    return disparity
