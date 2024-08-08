import os
from PIL import Image
from skimage.morphology import square, dilation
import yaml
from pathlib import Path
import numpy as np

import torch.utils.data
from .utils import numpy_sample_hints

class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'raw_mae': 'raw_mae.txt',
        'event': 'event',
        'raw': 'raw',
    }
    _DOMAIN = ['event', 'raw']
    NO_VALUE = 0.0

    def __init__(self, root, guideperc, randomperc, raw_frameratio, seq_size, raw_mae_threshold, rnd_sampling, dilation = 1):
        self.root = root
        self.guideperc = guideperc
        self.randomperc = randomperc
        self.raw_frameratio = raw_frameratio
        self.seq_size = seq_size
        self.raw_frameratio_counter = 0
        self.rnd_sampling = rnd_sampling
        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))
        self.dilation = dilation

        try:
            raw_maes = np.loadtxt(os.path.join(root, self._PATH_DICT['raw_mae']))
        except:
            raw_maes = None

        self.raw_maes = raw_maes

        self.disparity_path_list = {}
        self.timestamp_to_disparity_path = {}
        for domain in self._DOMAIN:
            
            if domain == 'raw':
                self.disparity_path_list[domain] = get_path_list(os.path.join(root, f"{self._PATH_DICT['raw']}"))
            else:
                self.disparity_path_list[domain] = get_path_list(os.path.join(root, self._PATH_DICT[domain]))


            self.timestamp_to_disparity_path[domain] = {timestamp: filepath for timestamp, filepath in
                                                        zip(self.timestamps, self.disparity_path_list[domain])}
        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_disparity_path['event'][timestamp]))[0])
            for timestamp in self.timestamp_to_disparity_path['event'].keys()
        }

        #Filter accordingly to arg and skip parameters
        if self.raw_maes is not None:
            tmp_timestamps = []
            for k in range(len(self.raw_maes)):
                if ((raw_mae_threshold is None or raw_mae_threshold <= 0 or self.raw_maes[k] <= raw_mae_threshold)):
                    tmp_timestamps.append(self.timestamps[k])
                
            self.timestamps = np.array(tmp_timestamps)

        if self.seq_size > 0:
            mysize = min(len(self.timestamps), self.seq_size)
            self.timestamps = self.timestamps[:mysize]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, timestamp):
        disp = load_disparity(self.timestamp_to_disparity_path['event'][timestamp], None)

        if self.raw_frameratio_counter % self.raw_frameratio == 0:
            if self.rnd_sampling:
                if self.randomperc:
                    guideperc = np.random.choice(range(0,6))*0.03
                else: 
                    guideperc = self.guideperc

                raw_disp = disp
                raw_disp = numpy_sample_hints(raw_disp, raw_disp>0, guideperc)
            else:
                raw_disp = load_disparity(self.timestamp_to_disparity_path['raw'][timestamp], None)
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
