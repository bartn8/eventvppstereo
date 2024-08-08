import numpy as np
import numba as nb
from numba.typed import List
from typing_extensions import Union, Tuple
import math

VALID_METHODS = ['h-naive', 'h-hist', 'h-voxelgrid', 'h-mdes', 'h-tore', 'h-timesurface']

def bth(left_history: Tuple[np.ndarray], right_history: Tuple[np.ndarray], gt: np.ndarray, occlusion_mask: Union[np.ndarray, None], guide_args: dict = None):
    """
    Insert fictitious events inside stereo event histories.
    Fictitious events are placed using geometry information given by some depth sensor.

    Parameters
    ----------
    left_history: tuple (x,y,p,t) of np.ndarray of shape (N,N,N,N)
        Left raw events. Assumptions: 0<=x<W and 0<=y<H and p in {0,1} and x,y are int32, p is int8, t is int64
    right_history: tuple (x,y,p,t) of np.ndarray of shape (N,N,N,N)
        Right raw events. Assumptions: 0<=x<W and 0<=y<H and p in {0,1}
    gt: nd.ndarray HxW
        Known sparse points. Zero for unkown values
    occlusion_mask: nd.ndarray HxW
        Known occluded points in target view. Zero for non-occluded and one for occluded.
    guide_args: dict
        Hyperparameters of hallucination:
        - 'bth_method': Choose a hallucination method based on event representation. Implemented methods: 'h-naive', 'h-hist', 'h-voxelgrid', 'h-mdes', 'h-tore', 'h-timesurface'
        - 'bth_n_events': Choose the number of fictitious events per pixel. Write a single number (e.g., '1') or a random range (e.g., '1:10') low and max included. Valid only for 'h-naive', 'h-hist', 'h-voxelgrid', 'h-mdes', 'h-timesurface'.
        - 'bth_injection_timestamp': Choose the timestamp where fictitious events will be placed. Valid values 'min', 'max', 'rnd'. Valid only for 'h-naive'
        - 'bth_splatting': Splat subpixel disparities between two adjacent pixels. Choices: 'none', 'spatial', 'temporal'.
        - 'bth_patch_size': choose spatial patch size (e.g., 1,3,5,7,9,...)
        - 'bth_uniform_patch': spatial patch with uniform polarity
        - 'bth_uniform_polarities': given a known point (X,Y) maintain same polarity in time domain
        - 'bth_n_bins': number of dense stacks of downstream event representation

    Return
    ------
    hallucinated_left_history: tuple (x,y,p,t) of np.ndarray of shape (N,N,N,N)
        Augmented left event history (x,y,p,t)
    hallucinated_right_history: tuple (x,y,p,t) of np.ndarray of shape (N,N,N,N)
        Augmented right event history (x,y,p,t)
    """

    left_x, left_y, left_p, left_t = left_history
    right_x, right_y, right_p, right_t = right_history

    assert left_x.dtype == np.int32 and left_y.dtype == np.int32 and right_x.dtype == np.int32 and right_y.dtype == np.int32
    assert left_p.dtype == np.int8 and right_p.dtype == np.int8
    assert left_t.dtype == np.int64 and right_t.dtype == np.int64

    left_x, left_y, left_p, left_t = np.copy(left_x.astype(np.int32)), np.copy(left_y.astype(np.int32)), np.copy(left_p.astype(np.int8)), np.copy(left_t.astype(np.int64))
    right_x, right_y, right_p, right_t = np.copy(right_x.astype(np.int32)), np.copy(right_y.astype(np.int32)), np.copy(right_p.astype(np.int8)), np.copy(right_t.astype(np.int64))

    left_history = (left_x, left_y, left_p, left_t)
    right_history = (right_x, right_y, right_p, right_t)

    if occlusion_mask is None:
        occlusion_mask = np.zeros_like(gt)   

    height, width = gt.shape[:2]
    min_t, max_t = left_t[0], left_t[-1]
    
    hallucination_method = guide_args['bth_method']
    raw_offset_us = 1

    if hallucination_method == 'h-naive':
        n_events = str(guide_args['bth_n_events'])
        injection_timestamp = guide_args['bth_injection_timestamp']
        splatting = guide_args['bth_splatting']
        wsize = guide_args['bth_patch_size']
        uniform_patch = guide_args['bth_uniform_patch']
        uniform_polarities = guide_args['bth_uniform_polarities']

        if ':' in n_events:
            min_value, max_value =  n_events.split(':')
            min_value, max_value = int(min_value), int(max_value)
            n_events_map = np.random.randint(min_value, max_value+1, (height, width))
        else:
            fill_value = int(n_events)
            n_events_map = np.full((height, width), fill_value)

        if injection_timestamp == 'min':
            injection_timestamp = min_t-raw_offset_us
        elif injection_timestamp == 'max':
            injection_timestamp = max_t-raw_offset_us
        elif injection_timestamp == 'rnd':
            injection_timestamp = _random_float_min_max(min_t, max(min_t, max_t-raw_offset_us))
        else:
            raise Exception(f"Invalid value for injection_timestamp: {injection_timestamp}")
        
        return _bth_naive(left_history, right_history, gt, occlusion_mask, n_events_map, injection_timestamp, splatting, wsize, uniform_patch, uniform_polarities)
    
    elif hallucination_method == 'h-hist':
        splatting = guide_args['bth_splatting']
        wsize = guide_args['bth_patch_size']
        uniform_patch = guide_args['bth_uniform_patch']
        n_bins = guide_args['bth_n_bins']
        max_n_events = str(guide_args['bth_n_events'])

        if ':' in max_n_events:
            min_value, max_value =  max_n_events.split(':')
            min_value, max_value = int(min_value), int(max_value)
            max_n_events = np.random.randint(min_value, max_value+1)
        else:
            max_n_events = int(max_n_events)

        return _bth_hist(left_history, right_history, gt, occlusion_mask, splatting, wsize, uniform_patch, n_bins, max_n_events, raw_offset_us=raw_offset_us)
    
    elif hallucination_method == 'h-voxelgrid':
        n_events = str(guide_args['bth_n_events'])
        splatting = guide_args['bth_splatting']
        wsize = guide_args['bth_patch_size']
        uniform_patch = guide_args['bth_uniform_patch']
        uniform_polarities = guide_args['bth_uniform_polarities']
        n_bins = guide_args['bth_n_bins']

        if ':' in n_events:
            min_value, max_value =  n_events.split(':')
            min_value, max_value = int(min_value), int(max_value)
            n_events_map = np.random.randint(min_value, max_value+1, (height, width))
        else:
            fill_value = int(n_events)
            n_events_map = np.full((height, width), fill_value)

        return _bth_voxel_grid(left_history, right_history, gt, occlusion_mask, n_events_map, splatting, wsize, uniform_patch, uniform_polarities, n_bins, raw_offset_us=raw_offset_us)
    
    elif hallucination_method == 'h-mdes':
        n_events = str(guide_args['bth_n_events'])
        splatting = guide_args['bth_splatting']
        wsize = guide_args['bth_patch_size']
        uniform_patch = guide_args['bth_uniform_patch']
        uniform_polarities = guide_args['bth_uniform_polarities']
        n_bins = guide_args['bth_n_bins']

        if ':' in n_events:
            min_value, max_value =  n_events.split(':')
            min_value, max_value = int(min_value), int(max_value)
            n_events_map = np.random.randint(min_value, max_value+1, (height, width))
        else:
            fill_value = int(n_events)
            n_events_map = np.full((height, width), fill_value)

        return _bth_mdes(left_history, right_history, gt, occlusion_mask, n_events_map, splatting, wsize, uniform_patch, uniform_polarities, n_bins, raw_offset_us=raw_offset_us)

    elif hallucination_method == 'h-tore':
        splatting = guide_args['bth_splatting']
        wsize = guide_args['bth_patch_size']
        uniform_patch = guide_args['bth_uniform_patch']
        n_bins = guide_args['bth_n_bins']

        return _bth_tore(left_history, right_history, gt, occlusion_mask, splatting, wsize, uniform_patch, n_bins, raw_offset_us=raw_offset_us)

    elif hallucination_method == 'h-timesurface':
        n_events = str(guide_args['bth_n_events'])
        splatting = guide_args['bth_splatting']
        wsize = guide_args['bth_patch_size']
        uniform_patch = guide_args['bth_uniform_patch']
        n_bins = guide_args['bth_n_bins']

        if ':' in n_events:
            min_value, max_value =  n_events.split(':')
            min_value, max_value = int(min_value), int(max_value)
            n_events_map = np.random.randint(min_value, max_value+1, (height, width))
        else:
            fill_value = int(n_events)
            n_events_map = np.full((height, width), fill_value)

        return _bth_time_surfaces(left_history, right_history, gt, occlusion_mask, n_events_map, splatting, wsize, uniform_patch, n_bins, raw_offset_us=raw_offset_us)

    else:
        raise Exception(f"{hallucination_method} is invalid. Choose between: {VALID_METHODS}")

@nb.njit
def _custom_insert(arr1, arr2, position):
    if len(arr2) > 0:
        result = np.empty(len(arr1) + len(arr2), dtype=arr1.dtype)

        for i in range(position):
            result[i] = arr1[i]
        
        for j,i in enumerate(range(position, position + len(arr2))):
            result[i] = arr2[j]
        
        for j,i in enumerate(range(position + len(arr2), len(arr1) + len(arr2))):
            result[i] = arr1[j+position]
        return result
    else:
        return arr1

@nb.njit
def _random_int_min_max(min,max):
    """
    Return random integers from min (inclusive) to max (exclusive).
    """
    if max - min > 1:
        return np.random.randint(min, max)
    else:
        return min

@nb.njit
def _random_float_min_max(min,max):
    """
    Return random float from min (inclusive) to max (exclusive).
    """
    return np.random.random() * (max-min) + min

@nb.njit
def _bth_naive_single(h_left_x: List, h_left_y: List, h_left_p: List, h_left_t: List, h_right_x: List, h_right_y: List, h_right_p: List, h_right_t: List, n: int, uniform_patch: bool, x: int, y: int, xd: int, height: int, width: int, injection_timestamp: Union[float, int], random_polarity: Union[None, np.ndarray] = None, rnd_min: int = 0, rnd_max: int = 2):
    if uniform_patch:
        if random_polarity is None:
            _random_polarity = _random_int_min_max(rnd_min,rnd_max)
        else:
            _random_polarity = random_polarity[0,0]

    min_n_y, max_n_y, min_n_x, max_n_x =  n,n,n,n

    for yw in range(-min_n_y,max_n_y+1):
        for xw in range(-min_n_x,max_n_x+1):
            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1:
                if not uniform_patch:
                    if random_polarity is None:
                        _random_polarity = _random_int_min_max(rnd_min,rnd_max)
                    else:
                        _random_polarity = random_polarity[yw+min_n_y, xw+min_n_x]

                if  0 <= xd+xw and xd+xw <= width-1:#  (1)
                    h_left_x.append(nb.int32(x+xw))
                    h_left_y.append(nb.int32(y+yw))
                    h_left_p.append(nb.int8(_random_polarity))
                    h_left_t.append(injection_timestamp)

                    h_right_x.append(nb.int32(xd+xw))
                    h_right_y.append(nb.int32(y+yw))
                    h_right_p.append(nb.int8(_random_polarity))
                    h_right_t.append(injection_timestamp)    
                else:
                    #Left side occlusion (known) (2)
                    h_left_x.append(nb.int32(x+xw))
                    h_left_y.append(nb.int32(y+yw))
                    h_left_p.append(nb.int8(_random_polarity))
                    h_left_t.append(injection_timestamp)
                    

@nb.njit
def _bth_naive(left_history: tuple, right_history: tuple, gt: np.ndarray, occlusion_mask: np.ndarray, n_events_map: np.ndarray, injection_timestamp: Union[float, int], splatting: str, wsize: int, uniform_patch: bool, uniform_polarities: bool, rnd_min: int = 0, rnd_max: int = 2):
    """
    Given a pair of rectified event history arrays, insert hallucinated events into the pair accordingly to known disparity points (of size M).
    All hallucinated events are inserted in the very same timestamp (sync) using a random polarity series.
    For example, given a known point in (X;Y), this method create n_events fake events such as [(X,Y,T,-1),(X,Y,T,1),(X,Y,T,1)...(X,Y,T,-1)].

    Parameters
    ----------

    left_history: tuple (x,y,p,t) of shape (N,N,N,N)
        Left original event history (x,y,p,t)
    right_history: tuple (x,y,p,t) of shape (N,N,N,N)
        Right original event history (x,y,p,t)
    gt: np.ndarray HxW
        Sparse disparity known points where H and W match sensor size
    n_events:  np.ndarray HxW
        Size of event series for each known (X,Y)
    injection_timestamp: float | int
        Timestamp used for each fake event
    splatting: str
        True to splat n_events events between two adjactent pixels


    Returns
    -------

    hallucinated_left_history: tuple (x,y,p,t) of shape (N+n_events*M,N+n_events*M,N+n_events*M,N+n_events*M)
        Augmented left event history (x,y,p,t)
    hallucinated_right_history: tuple (x,y,p,t) of shape (N+n_events*M,N+n_events*M,N+n_events*M,N+n_events*M)
        Augmented right event history (x,y,p,t)
    """

    height, width = gt.shape[:2]

    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    n = ((wsize -1) // 2)

    left_x, left_y, left_p, left_t = left_history
    right_x, right_y, right_p, right_t = right_history

    #Calculate temporal index for temporal splatting
    min_t, max_t = left_t[0], left_t[-1]
    range_t = (max_t-min_t)
    temporal_index = (injection_timestamp-min_t) / range_t

    #Calculate insertion index
    left_injection_idx = np.where(left_t<=injection_timestamp)[0]
    left_injection_idx = left_injection_idx[-1] if len(left_injection_idx) > 0 else 0
    right_injection_idx = np.where(right_t<=injection_timestamp)[0]
    right_injection_idx = right_injection_idx[-1] if len(right_injection_idx) > 0 else 0
   
    #Pre-calculating the number of added events is not trivial (patch size, splatting, left occlusions,...) so use lists
    h_left_x = List.empty_list(nb.int32)
    h_left_y = List.empty_list(nb.int32)
    h_left_p = List.empty_list(nb.int8)
    h_left_t = List.empty_list(nb.int64)

    h_right_x = List.empty_list(nb.int32)
    h_right_y = List.empty_list(nb.int32)
    h_right_p = List.empty_list(nb.int8)
    h_right_t = List.empty_list(nb.int64)
    
    #Create hallucinated events accordingly to sparse disparity points

    for y in range(height):
        for x in range(width):
            if gt[y,x] > 0 and occlusion_mask[y,x] == 0:#Discard occlued points
                d = round(gt[y,x])
                d0 = math.floor(gt[y,x]) 
                d1 = math.ceil(gt[y,x])  
                d1_blending = gt[y,x]-d0   
                d0_blending = 1-d1_blending
                
                #Splatting weights
                n_events = n_events_map[y,x]
                K_d1 = round(d1_blending * n_events)
                K_d0 = n_events-K_d1

                #Warping right (negative disparity hardcoded)
                xd = x-d
                xd0 = x-d0
                xd1 = x-d1

                if uniform_polarities:
                    if uniform_patch:
                        random_polarity = np.random.randint(rnd_min, rnd_max, (1,1))
                    else:
                        random_polarity = np.random.randint(rnd_min, rnd_max, (2*n+1,2*n+1))
                else:
                    random_polarity = None

                if splatting == 'spatial':
                    for _ in range(K_d0):
                        _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd0,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                    for _ in range(K_d1):
                        _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd1,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                elif splatting == 'temporal':
                    if d0_blending >= d1_blending:
                        if temporal_index < d1_blending:
                            for _ in range(n_events):
                                _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd1,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                        else:
                            for _ in range(n_events):
                                _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd0,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                    else:
                        if temporal_index < d0_blending:
                            for _ in range(n_events):
                                _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd0,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                        else:
                            for _ in range(n_events):
                                _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd1,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                else:
                    for _ in range(n_events):
                        _bth_naive_single(h_left_x,h_left_y,h_left_p,h_left_t,h_right_x,h_right_y,h_right_p,h_right_t,n,uniform_patch,x,y,xd,height,width,injection_timestamp,random_polarity,rnd_min,rnd_max)
                    
    return ((
            _custom_insert(left_x, h_left_x, left_injection_idx),
            _custom_insert(left_y, h_left_y, left_injection_idx),
            _custom_insert(left_p, h_left_p, left_injection_idx),
            _custom_insert(left_t, h_left_t, left_injection_idx)
        ),
        (
            _custom_insert(right_x, h_right_x, right_injection_idx),
            _custom_insert(right_y, h_right_y, right_injection_idx),
            _custom_insert(right_p, h_right_p, right_injection_idx),
            _custom_insert(right_t, h_right_t, right_injection_idx)
        ))

@nb.njit
def _bth_hist(left_history: tuple, right_history: tuple, gt: np.ndarray, occlusion_mask: np.ndarray, splatting: str, wsize: int, uniform_patch: bool, n_bins: int, max_n_events: int, raw_offset_us: int = 0):
    """
    2D Histogram aware event history hallucination.
    Using voxel grid hallucination to insert a different amount of k for each pixel
    Assumptions: No events in known points; number of bins known; left and right history temporally synchronized.
    """
    
    #Create two n_events_map (one for each polarity) accordingly to sparse input.
    height, width = gt.shape[:2]

    n_events_map_pos = np.zeros((height, width), dtype=np.uint32)
    n_events_map_neg = np.zeros((height, width), dtype=np.uint32)

    for y in range(height):
        for x in range(width):
            if gt[y,x] > 0:
                n_events_map_pos[y,x] = _random_int_min_max(0, max_n_events+1)
                n_events_map_neg[y,x] = _random_int_min_max(0, max_n_events+1)

    
    h_left_history_pos, h_right_history_pos = _bth_voxel_grid(left_history, right_history, gt, occlusion_mask, n_events_map_pos, splatting, wsize, uniform_patch, True, n_bins, 1, 2, raw_offset_us)
    return _bth_voxel_grid(h_left_history_pos, h_right_history_pos, gt, occlusion_mask, n_events_map_neg, splatting, wsize, uniform_patch, True, n_bins, 0, 1, raw_offset_us)

@nb.njit
def _bth_voxel_grid(left_history: tuple, right_history: tuple, gt: np.ndarray, occlusion_mask: np.ndarray, n_events_map: np.ndarray, splatting: str, wsize: int, uniform_patch: bool, uniform_polarities: bool, n_bins: int, rnd_min: int = 0, rnd_max: int = 2, raw_offset_us: int = 0):
    """
    Voxel Grid aware event history hallucination.
    Using naive hallucination to insert k event for each bin.
    Assumptions: number of bins known; left and right history temporally synchronized.
    """

    #Find the center of each bin
    _, _, _, left_t = left_history
    min_t, max_t = left_t[0], left_t[-1]
    range_t = (max_t-min_t)

    h_left_history, h_right_history = left_history, right_history

    #for each bin create k naive events
    for b in range(n_bins):
        left_ts = (b/n_bins) * range_t + min_t
        right_ts = ((b+1)/n_bins) * range_t + min_t
        center_ts = round((left_ts+right_ts)/2)
        center_ts = center_ts-raw_offset_us # Shift approach
        h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, n_events_map, center_ts, splatting, wsize, uniform_patch, uniform_polarities, rnd_min, rnd_max)

    return h_left_history, h_right_history

@nb.njit
def _bth_mdes(left_history: tuple, right_history: tuple, gt: np.ndarray, occlusion_mask: np.ndarray, n_events_map: np.ndarray, splatting: str, wsize: int, uniform_patch: bool, uniform_polarities: bool, n_bins: int, raw_offset_us: int = 0):

    """
    Mixed Density Event Stacking (SBT version) aware event history hallucination.
    Using naive hallucination to insert k event for each known point in a specific timestamp history.
    Assumptions: known number of stacks.
    """
    height, width = gt.shape[:2]
    _, _, _, left_t = left_history
    min_t, max_t = left_t[0], left_t[-1]
    len_t = (max_t-min_t)

    #Map to choose in which stack put each known point
    split_map = np.random.randint(0, n_bins, (height, width))

    h_left_history, h_right_history = left_history, right_history

    #for each bin-1 find correct timestamp and insert there fictitious events
    for b in range(n_bins-1):
        center_ts = round(((2**(b+1))-1)/(2**(b+1)) * len_t) + min_t
        center_ts = center_ts-raw_offset_us # Shift approach
        h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, np.multiply(np.where(split_map==b,1,0), n_events_map), center_ts, splatting, wsize, uniform_patch, uniform_polarities)

    #Last bin: insert in the most recent history
    h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, np.multiply(np.where(split_map==(n_bins-1),1,0), n_events_map), max_t, splatting, wsize, uniform_patch, uniform_polarities)

    return h_left_history, h_right_history

@nb.njit
def _bth_tore(left_history: tuple, right_history: tuple, gt: np.ndarray, occlusion_mask: np.ndarray, splatting: str, wsize: int, uniform_patch: bool, n_bins: int, raw_offset_us: int = 0):
    """
    Temporal Ordered Recent Events aware event history hallucination.
    Using naive hallucination to insert k event with different timestamps in last part of history.
    Assumptions: No events in known points; known number of stacks
    """

    #Only one event for bin
    height, width = gt.shape[:2]
    n_events_map = np.ones((height, width))

    #TORE use log(timestamp)
    half_n_bins = n_bins // 2
    _, _, _, left_t = left_history
    min_t, max_t = left_t[0], left_t[-1]
    min_t = min_t-raw_offset_us # Shift approach
    max_t = max_t-raw_offset_us # Shift approach
    len_t = (max_t-min_t)
    
    h_left_history, h_right_history = left_history, right_history

    tmp_min_t = min_t

    #for each bin-1 sample timestamp and insert there fictitious events
    for b in range(half_n_bins-1):
        tmp_max_t = (((2**(b+1))-1)/(2**(b+1)) * len_t) + min_t

        #Sample points from log distribution
        chosen_ts_pos = round(_random_float_min_max(tmp_min_t, tmp_max_t))
        chosen_ts_neg = round(_random_float_min_max(tmp_min_t, tmp_max_t))

        #Insert two events for each polarity
        h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, n_events_map, chosen_ts_pos, splatting, wsize, uniform_patch, True, 1, 2)
        h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, n_events_map, chosen_ts_neg, splatting, wsize, uniform_patch, True, 0, 1)

        tmp_min_t = tmp_max_t

    tmp_max_t = max_t

    #Last bin
    #Sample points from log distribution
    chosen_ts_pos = round(_random_float_min_max(tmp_min_t, tmp_max_t))
    chosen_ts_neg = round(_random_float_min_max(tmp_min_t, tmp_max_t))

    #Insert two events for each polarity
    h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, n_events_map, chosen_ts_pos, splatting, wsize, uniform_patch, True, 1, 2)
    h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, n_events_map, chosen_ts_neg, splatting, wsize, uniform_patch, True, 0, 1)

    return h_left_history, h_right_history

@nb.njit
def _bth_time_surfaces(left_history: tuple, right_history: tuple, gt: np.ndarray, occlusion_mask: np.ndarray, n_events_map: np.ndarray, splatting: str, wsize: int, uniform_patch: bool, n_bins: int, raw_offset_us: int = 0):
    """
    Time Surfaces aware event history hallucination.
    Using naive hallucination to insert k event with different timestamps in one or more timestamps.
    Assumptions: No events in known points; known number of stacks
    """

    height, width = gt.shape[:2]
    half_n_bins = n_bins // 2

    #ON-OFF map to choose in which stack(s) put each known point
    split_map_pos = np.random.randint(0, 2, (half_n_bins, height, width))
    split_map_neg = np.random.randint(0, 2, (half_n_bins, height, width))

    _, _, _, left_t = left_history
    min_t, max_t = left_t[0], left_t[-1]
    len_t = (max_t-min_t)

    h_left_history, h_right_history = left_history, right_history

    for b in range(half_n_bins):
        center_ts = round((((b+1)/half_n_bins) * len_t) + min_t)
        center_ts = center_ts-raw_offset_us # Shift approach
        h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, np.multiply(split_map_pos[b],n_events_map), center_ts, splatting, wsize, uniform_patch, True,1,2)
        h_left_history, h_right_history = _bth_naive(h_left_history, h_right_history, gt, occlusion_mask, np.multiply(split_map_neg[b],n_events_map), center_ts, splatting, wsize, uniform_patch, True,0,1)

    return h_left_history, h_right_history
