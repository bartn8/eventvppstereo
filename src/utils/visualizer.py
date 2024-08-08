import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def tensor_to_disparity_image(tensor_data):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    disparity_image = Image.fromarray(np.asarray(tensor_data * 256.0).astype(np.uint16))

    return disparity_image


def tensor_to_disparity_magma_image(tensor_data, vmax=None, mask=None):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    numpy_data = np.asarray(tensor_data)

    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)

    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_MAGMA)
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    if mask is not None:
        assert tensor_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)

    return disparity_image


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[np.floor(y[mask1]).astype(np.int32),np.floor(x[mask1]).astype(np.int32)]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img

_color_map_errors_kitti = np.array([
        [ 0,       0.1875, 149,  54,  49],
        [ 0.1875,  0.375,  180, 117,  69],
        [ 0.375,   0.75,   209, 173, 116],
        [ 0.75,    1.5,    233, 217, 171],
        [ 1.5,     3,      248, 243, 224],
        [ 3,       6,      144, 224, 254],
        [ 6,      12,       97, 174, 253],
        [12,      24,       67, 109, 244],
        [24,      48,       39,  48, 215],
        [48,  np.inf,       38,   0, 165]
]).astype(float)

def color_error_image_kitti(errors, scale=1, mask=None, BGR=True, dilation=1):
    errors_flat = errors.flatten()
    colored_errors_flat = np.zeros((errors_flat.shape[0], 3))
    
    for col in _color_map_errors_kitti:
        col_mask = np.logical_and(errors_flat>=col[0]/scale, errors_flat<=col[1]/scale)
        colored_errors_flat[col_mask] = col[2:]
        
    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 0

    if not BGR:
        colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]

    colored_errors = colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.uint8)

    if dilation>0:
        kernel = np.ones((dilation, dilation))
        colored_errors = cv2.dilate(colored_errors, kernel)
    return colored_errors