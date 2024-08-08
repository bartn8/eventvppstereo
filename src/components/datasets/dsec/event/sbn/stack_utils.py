import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import numba


@dataclass(frozen=True)
class ToTimesurface:
    """Create global or local time surfaces for each event. Modeled after the paper Lagorce et al.
    2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.
    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    """

    sensor_size: Tuple[int, int, int]
    surface_dimensions: Union[None, Tuple[int, int]] = None
    tau: float = 5e3
    decay: str = "lin"

    def __call__(self, events, indices):
        timestamp_memory = np.zeros(
            (self.sensor_size[2], self.sensor_size[1], self.sensor_size[0])
        )
        
        timestamp_memory -= self.tau * 3 + 1

        all_surfaces = np.zeros(
            (
                len(indices),
                self.sensor_size[2],
                self.sensor_size[1],
                self.sensor_size[0],
            )
        )

        to_timesurface_numpy(
            events["x"],
            events["y"],
            events["t"],
            events["p"],
            indices,
            timestamp_memory,
            all_surfaces,
            tau=self.tau,
        )
        return all_surfaces


@numba.jit(nopython=True)
def to_timesurface_numpy(x, y, t, p, indices, timestamp_memory, all_surfaces, tau=5e3):
    """Representation that creates timesurfaces for each event in the recording. Modeled after the
    paper Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern
    recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.
    Parameters:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    Returns:
        array of timesurfaces with dimensions (w,h) or (p,w,h)
    """
    current_index_pos = 0
    for index in range(len(x)):
        timestamp_memory[p[index], y[index], x[index]] = t[index]

        if index == indices[current_index_pos]:
            #Time surface is causal: surface is created based only with timestamps <= t[index]
            timestamp_context = timestamp_memory - t[index]
            all_surfaces[current_index_pos, :, :, :] = np.exp(timestamp_context / tau)
            current_index_pos += 1
            if current_index_pos > len(indices) - 1:
                break

@numba.jit(nopython=True)
def _fast_queue(tore, px, py, pts_diff):
    for t, i, j in zip(pts_diff, py, px):
        #Assume ordered ts
        tore[i - 1, j - 1] = np.roll(tore[i - 1, j - 1], 1)
        tore[i - 1, j - 1, 0] = t
    

def events2ToreFeature(x, y, ts, pol, sampleTimes, k, frameSize):
    oldPosTore = np.inf * np.ones((frameSize[0], frameSize[1], 2 * k))# HxWxC (see stack.py framesize format)
    oldNegTore = np.inf * np.ones((frameSize[0], frameSize[1], 2 * k))# HxWxC (see stack.py framesize format)
    
    Xtore = np.zeros((frameSize[0], frameSize[1], 2 * k, 1), dtype=np.float32)# HxWxCx1 (see stack.py framesize format)

    priorSampleTime = -np.inf

    sampleLoop, currentSampleTime = 0, sampleTimes

    addEventIdx = np.logical_and(ts >= priorSampleTime, ts <= currentSampleTime)

    p = np.logical_and(addEventIdx, pol > 0)
    px, py, pts_diff = x[p], y[p], currentSampleTime - ts[p]

    newPosTore = np.full(frameSize + (k,), np.inf)#HxWxK

    _fast_queue(newPosTore, px, py, pts_diff)

    p = np.logical_and(addEventIdx, pol <= 0)
    px, py, pts_diff = x[p], y[p], currentSampleTime - ts[p]

    newNegTore = np.full(frameSize + (k,), np.inf)#HxWxK

    _fast_queue(newNegTore, px, py, pts_diff)
    
    oldPosTore += currentSampleTime - priorSampleTime
    oldPosTore = np.dstack([oldPosTore[..., :k], newPosTore]).reshape(frameSize[0], frameSize[1], 2, -1).min(axis=2)#dstack(HxWxK, HxWxK) -> reshape(HxWx2xK) -> min -> HxWxK

    oldNegTore += currentSampleTime - priorSampleTime
    oldNegTore = np.dstack([oldNegTore[..., :k], newNegTore]).reshape(frameSize[0], frameSize[1], 2, -1).min(axis=2)#dstack(HxWxK, HxWxK) -> reshape(HxWx2xK) -> min -> HxWxK
    
    Xtore[:, :, :, sampleLoop] = np.concatenate([oldPosTore, oldNegTore], axis=2).astype(np.float32)#cat(HxWxK,HxWxK) -> HxWx(2*K=C)

    priorSampleTime = currentSampleTime

    minTime = 150
    maxTime = 500e6

    tmp = Xtore[:, :, :, 0]
    tmp[np.isnan(tmp)] = maxTime
    tmp[tmp > maxTime] = maxTime
    tmp = np.log(tmp + 1)
    tmp -= np.log(minTime + 1)
    tmp[tmp < 0] = 0
    Xtore[:, :, :, 0] = tmp

    return Xtore[
        ..., 0
    ]  # (H, W, C, 1) to (H, W, C)
