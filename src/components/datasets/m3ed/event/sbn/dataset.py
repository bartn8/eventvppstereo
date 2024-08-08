import os
import numpy as np
import torch.utils.data
import cv2
import zlib
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .slice import EventSlicer
from . import stack, constant

from .bth import bth
from .filter import occlusion_heuristic

def render(img: np.ndarray, x: np.ndarray, y: np.ndarray, pol: np.ndarray, t: np.ndarray, color_neg = (255,0,0,255), color_pos = (0,0,255,255), scale_factor: int = 1) -> np.ndarray:
    
    #1: Vanilla negative polarity Blue #0000ff
    #2: Vanilla positive polarity Red #ff0000
    #3: Fictitious negative polarity magenta #ff00ff
    #4: Fictitious positive polarity Lime #00ff00

    #Use BGRA instead

    assert x.size == y.size == pol.size == t.size
    N_BINS,H,W = img.shape[:3]
    H,W = H*scale_factor,W*scale_factor
    
    #img = np.full((H,W,3), fill_value=255,dtype='uint8')
    min_t, max_t = t[0], t[-1]
    range_t = (max_t-min_t)

    ts_prev = 0

    for b in range(N_BINS):
        ts = ((b+1)/N_BINS) * range_t + min_t

        mask = np.zeros((H//scale_factor,W//scale_factor),dtype='int32')
        pol = pol.astype('int')#implicit copy
        pol[pol==0]=-1

        mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)&(ts_prev<=t)&(t<=ts)
        mask[np.floor(y[mask1]//scale_factor).astype(np.int32),np.floor(x[mask1]//scale_factor).astype(np.int32)]=pol[mask1]

        img[b][(mask==-1)] = color_neg 
        img[b][(mask== 1)] = color_pos

        ts_prev = ts

    return img

class EventDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'timestamp': 'timestamps.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = None

    def __init__(self, root, num_of_event, stack_method, stack_size,
                 num_of_future_event=0, use_preprocessed_image=False, normalize=True, use_compression=False, **kwargs):
        self.root = root
        self.seq_name = root.split("/")[-2]
        self.num_of_event = num_of_event
        self.stack_method = stack_method
        self.stack_size = stack_size
        self.num_of_future_event = num_of_future_event
        self.use_preprocessed_image = use_preprocessed_image
        self.normalize = normalize
        self.use_compression = use_compression
        
        self.event_slicer = {}
        for location in self._LOCATION:
            event_path = os.path.join(root, location, 'events.h5')
            rectify_map_path = os.path.join(root, location, 'rectify_map.h5')
            self.event_slicer[location] = EventSlicer(event_path, rectify_map_path, num_of_event, num_of_future_event)

        self.stack_function = getattr(stack, stack_method)(stack_size, num_of_event,
                                                           constant.EVENT_HEIGHT, constant.EVENT_WIDTH, normalize, **kwargs)
        self.NO_VALUE = self.stack_function.NO_VALUE

        self.mycounter = 0

    def __len__(self):
        return 0

    def __getitem__(self, mydict):
        timestamp, hints, guide_args = mydict['timestamp'], mydict['hints'], mydict['guide_args']

        if self.use_preprocessed_image and hints is None:
            data_define = 'sbn_%d_%s_%d_%d_%d' % (self.num_of_event, self.stack_method, (1 if self.normalize else 0), self.stack_size, self.num_of_future_event)
            save_root = os.path.join(self.root, data_define)
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, '%ld.npy' % timestamp)
            if os.path.exists(save_path):
                #Check for exceptions in the file
                try:
                    if self.use_compression:
                        with open(save_path, "rb") as f:
                            buffer = io.BytesIO(zlib.decompress(f.read()))
                        event_data = torch.load(buffer)
                        
                        del buffer
                    else:
                        event_data = torch.load(save_path)

                except:
                    #PytorchStreamReader failed reading zip archive: failed finding central directory
                    #Recompile and save
                    event_data = self._pre_load_event_data(timestamp=timestamp, hints=None, guide_args=guide_args)
                    try:
                        os.remove(save_path)
                        if self.use_compression:
                            buffer = io.BytesIO()
                            torch.save(event_data, buffer)
                            with open(save_path, "wb") as f:
                                f.write(zlib.compress(buffer.getbuffer()))
                            
                            del buffer
                        else:
                            torch.save(event_data, save_path)    
                    except:
                        pass
            else:
                event_data = self._pre_load_event_data(timestamp=timestamp, hints=None, guide_args=guide_args)
                try:
                    if self.use_compression:
                        buffer = io.BytesIO()
                        torch.save(event_data, buffer)
                        with open(save_path, "wb") as f:
                            f.write(zlib.compress(buffer.getbuffer()))
                        
                        del buffer
                    else:
                        torch.save(event_data, save_path)
                except:
                    pass
        else:
            event_data = self._pre_load_event_data(timestamp=timestamp, hints=hints, guide_args=guide_args)

        event_data = self._post_load_event_data(event_data)

        return event_data

    def _pre_load_event_data(self, timestamp, hints, guide_args):
        event_data = {}
        fake_event_data = None

        if guide_args is not None:
            guide_method = guide_args['guide_method']
            guide_args['stack_method'] = self.stack_method
        else:
            guide_method = 'none'

        minimum_time, maximum_time = -float('inf'), float('inf')
        for location in self._LOCATION:
            event_data[location] = self.event_slicer[location][timestamp]
            minimum_time = max(minimum_time, event_data[location]['t'].min())
            maximum_time = min(maximum_time, event_data[location]['t'].max())

        for location in self._LOCATION:
            mask = np.logical_and(minimum_time <= event_data[location]['t'], event_data[location]['t'] <= maximum_time)
            for data_type in ['x', 'y', 't', 'p']:
                event_data[location][data_type] = event_data[location][data_type][mask]

        left_history = event_data['left']['x'].astype(np.int32), event_data['left']['y'].astype(np.int32), event_data['left']['p'].astype(np.int8), event_data['left']['t'].astype(np.int64)
        right_history = event_data['right']['x'].astype(np.int32), event_data['right']['y'].astype(np.int32), event_data['right']['p'].astype(np.int8), event_data['right']['t'].astype(np.int64)

        if hints is not None:
            occ_mask = occlusion_heuristic(hints) if guide_args['bth_maskocc'] else None
            #Convert to tuples, get occ mask, do hallucination and replace previous data structure.
            if 'bth' in guide_method:
                left_history = event_data['left']['x'].astype(np.int32), event_data['left']['y'].astype(np.int32), event_data['left']['p'].astype(np.int8), event_data['left']['t'].astype(np.int64)
                right_history = event_data['right']['x'].astype(np.int32), event_data['right']['y'].astype(np.int32), event_data['right']['p'].astype(np.int8), event_data['right']['t'].astype(np.int64)
                h_left_history, h_right_history = bth(left_history, right_history, hints, occ_mask, guide_args)

                #Use fused event history
                for location, history in zip(self._LOCATION, [h_left_history, h_right_history]):
                    for i, data_type in enumerate(['x', 'y', 'p', 't']):
                        event_data[location][data_type] = history[i]
        
        if guide_args['render']:
            n_images = 1
            scale_factor = 1
            
            img_left = np.full((n_images, constant.EVENT_HEIGHT//scale_factor, constant.EVENT_WIDTH//scale_factor, 4), 255, dtype='uint8')
            img_right = np.full((n_images, constant.EVENT_HEIGHT//scale_factor, constant.EVENT_WIDTH//scale_factor, 4), 255, dtype='uint8')
            img_left = render(img_left, event_data['left']['x'].astype(np.int32), event_data['left']['y'].astype(np.int32), event_data['left']['p'].astype(np.int8), event_data['left']['t'].astype(np.int64), scale_factor=scale_factor)
            img_right = render(img_right, event_data['right']['x'].astype(np.int32), event_data['right']['y'].astype(np.int32), event_data['right']['p'].astype(np.int8), event_data['right']['t'].astype(np.int64), scale_factor=scale_factor)

            rendered_data = {
                    'eh_left': img_left,
                    'eh_right': img_right,
                    }
            
            if 'bth' in guide_method:
                img_left_h = np.full((n_images, constant.EVENT_HEIGHT//scale_factor, constant.EVENT_WIDTH//scale_factor, 4), 255, dtype='uint8')
                img_right_h = np.full((n_images, constant.EVENT_HEIGHT//scale_factor, constant.EVENT_WIDTH//scale_factor, 4), 255, dtype='uint8')

                img_left_h = render(img_left_h, h_left_history[0], h_left_history[1], h_left_history[2], h_left_history[3], (255,0,255,255), (0,255,0,255), scale_factor)
                img_left_h = render(img_left_h, left_history[0], left_history[1], left_history[2], left_history[3], scale_factor=scale_factor)
                img_right_h = render(img_right_h, h_right_history[0], h_right_history[1], h_right_history[2], h_right_history[3], (255,0,255,255), (0,255,0,255), scale_factor)
                img_right_h = render(img_right_h, right_history[0], right_history[1], right_history[2], right_history[3], scale_factor=scale_factor)

                rendered_data['eh_left_h'] = img_left_h
                rendered_data['eh_right_h'] = img_right_h
                

            for key in rendered_data:    
                if n_images > 1:                
                    images=[]
                    for i in range(n_images):
                        images.append(Image.fromarray(cv2.cvtColor(rendered_data[key][i], cv2.COLOR_BGRA2RGB)))

                    images[0].save(os.path.join(guide_args['savedir'], f'{self.seq_name}_{self.mycounter:05d}_{key}.gif'), save_all=True, append_images=images[1:], duration=100, loop=0)
                else:
                    mypath = os.path.join(guide_args['savedir'], f'{self.seq_name}_{self.mycounter:05d}_{key}.jpg')
                    cv2.imwrite(mypath, cv2.cvtColor(rendered_data[key][0], cv2.COLOR_BGRA2BGR))
            self.mycounter += 1
            
        for location in self._LOCATION:
            event_data[location] = self.stack_function.pre_stack(event_data[location], timestamp)

        if fake_event_data is not None:
            for location in self._LOCATION:
                fake_event_data[location] = self.stack_function.pre_stack(fake_event_data[location], timestamp)
        
            return event_data, fake_event_data

        return event_data

    def _post_load_event_data(self, event_data):
        if isinstance(event_data, (tuple, list)):
            event_data, fake_event_data = event_data
            
            for location in self._LOCATION:
                event_data[location] = self.stack_function.post_stack(event_data[location])#HxWx{1,2}xCx1
                fake_event_data[location] = self.stack_function.post_stack(fake_event_data[location])#HxWx{1,2}xCx1
                event_data[location] = np.concatenate([event_data[location], fake_event_data[location]], axis=3)#HxWx{1,2}x(2*C)x1

        elif isinstance(event_data, dict):
            for location in self._LOCATION:
                event_data[location] = self.stack_function.post_stack(event_data[location])#HxWx{1,2}xCx1
    
        return event_data

    def collate_fn(self, batch):
        batch = self.stack_function.collate_fn(batch)

        return batch
