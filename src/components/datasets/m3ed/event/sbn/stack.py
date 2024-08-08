import torch
import numpy as np
from abc import ABC, abstractmethod
from .stack_utils import events2ToreFeature, ToTimesurface
from .mixed_density_event_stack import MixedDensityEventStack

class EventStacking(ABC):
    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch

    def __init__(self, stack_size, num_of_event, height, width, normalize, cache_filling = 0):
        self.stack_size = stack_size
        self.num_of_event = num_of_event
        self.height = height
        self.width = width
        self.normalize = normalize
        self.cache_filling = cache_filling

    @abstractmethod
    def make_stack(self, x, y, p, t):
        pass

    @abstractmethod
    def stack_data(self, x, y, p, t_s):
        pass

    @abstractmethod
    def make_empty_stack(self):
        pass

    def pre_stack(self, event_sequence, last_timestamp):
        x = event_sequence['x'].astype(np.int32)
        y = event_sequence['y'].astype(np.int32)
        p = 2 * event_sequence['p'].astype(np.int8) - 1
        t = event_sequence['t'].astype(np.int64)

        assert len(x) == len(y) == len(p) == len(t)

        past_mask = t < last_timestamp
        if np.sum(past_mask) == 0:
            past_stacked_event = self.make_empty_stack()
        else:
            p_x, p_y, p_p, p_t = x[past_mask], y[past_mask], p[past_mask], t[past_mask]
            p_t = p_t - p_t.min()
            past_stacked_event = self.make_stack(p_x, p_y, p_p, p_t)

        stacked_event_list = [past_stacked_event]

        return stacked_event_list

    def post_stack(self, pre_stacked_event):
        stacked_event_list = []

        for pf_stacked_event in pre_stacked_event:
            cur_stacked_event_list = []

            for stack_idx in range(self.stack_size - 1, -1, -1):
                stacked_data = np.full([self.height, self.width, 1], self.cache_filling, dtype=np.float32)#HxWx1
                #Filter using index mask
                stacked_data.put(pf_stacked_event['index'][stack_idx],
                                     pf_stacked_event['stacked_data'][stack_idx])
                cur_stacked_event_list.append(np.stack([stacked_data], axis=2))#unsqueeze(2) -> #[HxWx1x1, HxWx1x1, ...]

            stacked_event_list.append(np.concatenate(cur_stacked_event_list[::-1], axis=2))#HxWxCx1

        if len(stacked_event_list) == 2:#[past_stacked_event, future_stacked_event]
            stacked_event_list[1] = stacked_event_list[1][:, :, ::-1, :]

        stacked_event = np.stack(stacked_event_list, axis=2)#unsqueeze(2) -> HxWx{1,2}xCx1

        return stacked_event

class MixedDensityEventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(stack_size, num_of_event, height, width, normalize) 
          
    def make_stack(self, x, y, p, t):
        # t = t - t.min()
        # time_interval = t.max() - t.min() + 1
        # t_s = (t / time_interval * 2) - 1.0
        t_s = t
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        cur_num_of_events = len(t)
        for _ in range(self.stack_size):
            stacked_event = self.stack_data(x, y, p, t_s)
            stacked_event_list['stacked_data'].append(stacked_event['stacked_data'])

            cur_num_of_events = cur_num_of_events // 2
            x = x[cur_num_of_events:]
            y = y[cur_num_of_events:]
            p = p[cur_num_of_events:]
            t_s = t_s[cur_num_of_events:]
            t = t[cur_num_of_events:]

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0
            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        stacked_data = np.zeros([self.height, self.width], dtype=np.int8)

        index = (y * self.width) + x

        stacked_data.put(index, p)

        stacked_event = {
            'stacked_data': stacked_data,
        }

        return stacked_event
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        for _ in range(self.stack_size):
            stacked_event_list['stacked_data'].append(np.zeros([self.height, self.width], dtype=np.int8))
        
        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0
            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

class HistogramEventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(2, num_of_event, height, width, normalize) 

    def make_stack(self, x, y, p, t):
        t_s = t
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event = self.stack_data(x, y, p, t_s)
        stacked_event_list['stacked_data'].append(stacked_event['stacked_data_pos'])
        stacked_event_list['stacked_data'].append(stacked_event['stacked_data_neg'])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0
            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        stacked_data_pos = np.zeros([self.height, self.width], dtype=np.float32)
        flatten_stacked_data_pos = stacked_data_pos.ravel()
        stacked_data_neg = np.zeros([self.height, self.width], dtype=np.float32)
        flatten_stacked_data_neg = stacked_data_neg.ravel()

        pos_mask = p > 0
        neg_mask = p < 0

        index_pos = (y[pos_mask] * self.width) + x[pos_mask]
        index_neg = (y[neg_mask] * self.width) + x[neg_mask]
        
        np.add.at(flatten_stacked_data_pos, index_pos, 1)
        np.add.at(flatten_stacked_data_neg, index_neg, 1)

        stacked_data_pos = np.reshape(flatten_stacked_data_pos, (self.height, self.width))
        stacked_data_neg = np.reshape(flatten_stacked_data_neg, (self.height, self.width))

        if self.normalize:#Channel normalization
            stacked_data_pos = (stacked_data_pos-stacked_data_pos.min()) / (stacked_data_pos.max()-stacked_data_pos.min())
            stacked_data_neg = (stacked_data_neg-stacked_data_neg.min()) / (stacked_data_neg.max()-stacked_data_neg.min())

        stacked_event = {
            'stacked_data_pos': stacked_data_pos,
            'stacked_data_neg': stacked_data_neg,
        }

        return stacked_event
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event_list['stacked_data'].append(np.zeros([self.height, self.width], dtype=np.float32))
        stacked_event_list['stacked_data'].append(np.zeros([self.height, self.width], dtype=np.float32))

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0
            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list
    
class VoxelGridEventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(stack_size, num_of_event, height, width, normalize) 

    def make_stack(self, x, y, p, t):
        time_interval = t.max() - t.min()
        t_s = ((t.astype(np.float32) - t.min()) / time_interval) * (self.stack_size-1)
        
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event = self.stack_data(x, y, p, t_s)

        for i in range(stacked_event['stacked_data'].shape[0]):
            stacked_event_list['stacked_data'].append(stacked_event['stacked_data'][i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)
        # polarity should be +1 / -1

        voxel_grid = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)
        flatten_voxel_grid = voxel_grid.ravel()

        #get decimal part from ts
        int_t_s = t_s.astype(int)
        dec_t_s = t_s-int_t_s

        #Use decimal part to weight to two adjacent bins
        vals_left = p * (1.0-dec_t_s)
        vals_right = p * (dec_t_s)

        valid_indices = int_t_s < self.stack_size
        np.add.at(
            flatten_voxel_grid,
            #H and W indices with x,y and int_t_s for channel
            (x[valid_indices] + y[valid_indices] * self.width) + (int_t_s[valid_indices] * self.width * self.height),
            vals_left[valid_indices],
        )

        valid_indices = (int_t_s + 1) < self.stack_size
        np.add.at(
            flatten_voxel_grid,
            #H and W indices with x,y and int_t_s for channel
            (x[valid_indices] + y[valid_indices] * self.width) + ((int_t_s[valid_indices] + 1) * self.width * self.height),
            vals_right[valid_indices],
        )

        voxel_grid = np.reshape(
            flatten_voxel_grid, (self.stack_size, self.height, self.width)
        )

        if self.normalize:
            #Channel-wise normalization CxHxW
            min_values = voxel_grid.min(axis=(1, 2), keepdims=True)
            max_values = voxel_grid.max(axis=(1, 2), keepdims=True)

            # Channel-wise normalization between 0 and 1
            voxel_grid = (voxel_grid - min_values) / (max_values - min_values)
            #Map between -1 and 1
            voxel_grid = (voxel_grid * 2) - 1

        stacked_event = {
            'stacked_data': voxel_grid,
        }

        return stacked_event
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        voxel_grid = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)

        for i in range(voxel_grid.shape[0]):
            stacked_event_list['stacked_data'].append(voxel_grid[i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list
       
class ToreEventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(stack_size, num_of_event, height, width, normalize, 1)

    def make_stack(self, x, y, p, t):
        t_s = t
        
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event = self.stack_data(x, y, p, t_s)

        for i in range(stacked_event['stacked_data'].shape[0]):
            stacked_event_list['stacked_data'].append(stacked_event['stacked_data'][i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 1

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert self.stack_size % 2 == 0
        half_stack_size = self.stack_size // 2
        x = x + 1
        y = y + 1
        sampleTimes = t_s[-1]
        frameSize = (self.height, self.width)

        rep = events2ToreFeature(x, y, t_s, p, sampleTimes, half_stack_size, frameSize)#HxWxC
        #result: CxHxW or something like that
        rep = np.transpose(rep, (2,0,1))

        if self.normalize:
            #Group-wise (pos/neg polarity) normalization
            rep[:half_stack_size] = (rep[:half_stack_size]-rep[:half_stack_size].min()) / (rep[:half_stack_size].max()-rep[:half_stack_size].min())
            rep[half_stack_size:] = (rep[half_stack_size:]-rep[half_stack_size:].min()) / (rep[half_stack_size:].max()-rep[half_stack_size:].min())

        stacked_event = {
            'stacked_data': rep,
        }

        return stacked_event
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        
        minTime = 150
        maxTime = 500e6
        tore_tensor = np.full((self.stack_size, self.height, self.width), maxTime)
        tore_tensor = np.log(tore_tensor + 1)
        tore_tensor -= np.log(minTime + 1)
        tore_tensor[tore_tensor < 0] = 0

        if self.normalize:
            tore_tensor = np.ones((self.stack_size, self.height, self.width))

        for i in range(tore_tensor.shape[0]):
            stacked_event_list['stacked_data'].append(tore_tensor[i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 1

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list
        

class TimeSurfaceEventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(stack_size, num_of_event, height, width, normalize) 
        self.to_timesurface = ToTimesurface(
            sensor_size=(width, height, 2),
            surface_dimensions=None,
            tau=50000,
            decay="exp",
        )

    def make_stack(self, x, y, p, t):
        t_s = t

        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event = self.stack_data(x, y, p, t_s)

        for i in range(stacked_event['stacked_data'].shape[0]):
            stacked_event_list['stacked_data'].append(stacked_event['stacked_data'][i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t):
        assert self.stack_size % 2 == 0
        half_stack_size = self.stack_size // 2

        p = ((p+1)/2).astype(np.int8)
        t_norm = (t - t.min()) / (t.max() - t.min()) * half_stack_size
        idx = np.searchsorted(t_norm, np.arange(half_stack_size) + 1)#np.arange(half_stack_size) + 1 -> [1,2,3,4...,half_stack_size]
        rep = self.to_timesurface({'x':x,'y':y,'p':p,'t':t}, idx)# (idx)x2xHxW

        if self.normalize:
            #Group-wise normalization based on polarity
            rep[:,0,...] = (rep[:,0,...]-rep[:,0,...].min()) / (rep[:,0,...].max()-rep[:,0,...].min())
            rep[:,1,...] = (rep[:,1,...]-rep[:,1,...].min()) / (rep[:,1,...].max()-rep[:,1,...].min())

        rep = rep.reshape((-1, rep.shape[-2], rep.shape[-1]))#(idx*2=C)xHxW
        #CxHxW or something like that
        stacked_event = {
            'stacked_data': rep,
        }

        return stacked_event
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        timesurface_tensor = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)

        for i in range(timesurface_tensor.shape[0]):
            stacked_event_list['stacked_data'].append(timesurface_tensor[i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

class ERGO12EventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(12, num_of_event, height, width, normalize) 

    def make_stack(self, x, y, p, t):
        t_s = t

        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event = self.stack_data(x, y, p, t_s)

        for i in range(stacked_event['stacked_data'].shape[0]):
            stacked_event_list['stacked_data'].append(stacked_event['stacked_data'][i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t):
        window_indexes = [0, 3, 2, 6, 5, 6, 2, 5, 1, 0, 4, 1]
        functions = ["polarity", "timestamp_neg", "count_neg", "polarity", "count_pos", "count", "timestamp_pos", "count_neg", "timestamp_neg", "timestamp_pos", "timestamp", "count",]
        aggregations = [ "variance", "variance", "mean", "sum", "mean", "sum", "mean", "mean", "max", "max", "max", "mean",]

        stacking_type = ["SBN", "SBT"][0]  # stacking based on number of events (SBN) or time (SBT)

        indexes_functions_aggregations = window_indexes, functions, aggregations

        transformation = MixedDensityEventStack( self.stack_size, self.num_of_event, self.height, self.width, indexes_functions_aggregations, stacking_type,)

        rep = transformation.stack({'x':x,'y':y,'p':p,'t':t})#HxWxC

        #CxHxW or something like that
        rep = np.transpose(rep, (2,0,1))

        if self.normalize:
            #Channel-wise normalization CxHxW
            min_values = rep.min(axis=(1, 2), keepdims=True)#Cx1x1
            max_values = rep.max(axis=(1, 2), keepdims=True)#Cx1x1

            # Channel-wise normalization between 0 and 1
            myrange = max_values - min_values#Cx1x1
            rep = (rep - min_values)#CxHxW

            for j in range(myrange.shape[0]):
                if myrange[j,0,0] > 0:
                    rep[j,:,:] = rep[j,:,:] / myrange[j,0,0]

            #rep = rep / myrange 

        stacked_event = {
            'stacked_data': rep,
        }

        return stacked_event
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        ergo12_tensor = np.zeros([self.stack_size, self.height, self.width], dtype=np.float32)

        for i in range(ergo12_tensor.shape[0]):
            stacked_event_list['stacked_data'].append(ergo12_tensor[i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list
        
class TencodeEventStacking(EventStacking):
    NO_VALUE = 0.
    STACK_LIST = ['stacked_data', 'index']

    def __init__(self, stack_size, num_of_event, height, width, normalize):
        super().__init__(3, num_of_event, height, width, normalize) 


    def make_stack(self, x, y, p, t):
        t = t - t.min()
        time_interval = t.max() - t.min()
        t_s = (t / time_interval)
        
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        stacked_event = self.stack_data(x, y, p, t_s)

        for i in range(stacked_event['stacked_data'].shape[0]):
            stacked_event_list['stacked_data'].append(stacked_event['stacked_data'][i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        #t_s normalized between [0,1]
        p = ((p+1)/2) #0:neg,1:pos

        rep = np.zeros((3,self.height, self.width), dtype=np.float32)
        
        index_red = (0 * self.width * self.height) + (y * self.width) + x
        index_green = (1 * self.width * self.height) + (y * self.width) + x
        index_blue = (2 * self.width * self.height) + (y * self.width) + x

        rep.put(index_red, 255*p)
        rep.put(index_green, 255*(1-t_s))
        rep.put(index_blue, 255*(1-p))

        if self.normalize:
            # normalization between 0 and 1
            rep = rep / 255

        stacked_event = {
            'stacked_data': rep,
        }

        return stacked_event#CxHxW 
    
    def make_empty_stack(self):
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}

        tencode_tensor = np.zeros([3, self.height, self.width], dtype=np.float32)

        for i in range(tencode_tensor.shape[0]):
            stacked_event_list['stacked_data'].append(tencode_tensor[i])

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        
        for stack_idx in range(self.stack_size):

            stack_data = stacked_event_list['stacked_data'][stack_idx]

            mask = stack_data != 0

            #From 2D (X,Y) to Flatten and use index preserve coordinate information
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_data'][stack_idx] = stack_data[mask]

        return stacked_event_list        
