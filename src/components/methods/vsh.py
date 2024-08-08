import numpy as np
import math
import numba as nb

@nb.njit
def _random_float_min_max(min,max):
    """
    Return random float from min (inclusive) to max (exclusive).
    """
    return np.random.random() * (max-min) + min

@nb.njit
def _virtual_projection_scan_rnd(left, right, gt, uniform_color, wsize, direction, blending, blending_occ, occlusion_mask, discard_occ, splatting, minvalue, maxvalue):

    """
    Virtual projection using sparse disparity.

    Parameters
    ----------
    left: np.numpy [H,W,C] 
        Left original image
    right: np.numpy [H,W,C] 
        Right original image        
    gt: np.numpy [H,W] 
        Sparse disparity
    uniform_color: bool
        True to project a uniform color patch
    wsize: int
        Max projection patch size (Default 5)     
    direction: int mod 2
        Projection direction (left->right or right->left) (Default 0)
    blending: float
        alpha blending factor
    blending_occ: float
        alpha blending factor in occluded areas
    occlusion_mask: np.numpy [H,W] 
        Occlusion mask (If not present use np.zeros(left.shape, dtype=))
    discard_occ: bool
        True to discard occluded points
    splatting: bool
        True to activate splatting between two adjacent pixels

    Returns
    -------
    sample_i:
        number of points projected
    """

    sample_i = 0

    height, width, channels = left.shape[:3]
    
    #Window size factor (wsize=2*n+1 -> n = (wsize-1)/2)
    n = ((wsize -1) // 2)
    min_n_y, max_n_y, min_n_x, max_n_x =  n,n,n,n

    for y in range(height):
        x = width-1 if direction == 0 else 0

        while (direction != 0 and x < width) or (direction == 0 and x>=0):
            if gt[y,x] > 0:
                d = round(gt[y,x])
                d0 = math.floor(gt[y,x])
                d1 = math.ceil(gt[y,x])  
                d1_blending = gt[y,x]-d0   

                #Warping right (negative disparity hardcoded)
                xd = x-d
                xd0 = x-d0
                xd1 = x-d1
                
                for j in range(channels):

                    if uniform_color:
                        rvalue = _random_float_min_max(minvalue, maxvalue)

                    for yw in range(-min_n_y,max_n_y+1):
                        for xw in range(-min_n_x,max_n_x+1):
                            if 0 <= y+yw and y+yw <= height-1 and 0 <= x+xw and x+xw <= width-1:                                     

                                if not uniform_color:
                                    rvalue = _random_float_min_max(minvalue, maxvalue)

                                if  0 <= xd0+xw and xd0+xw <= width-1:#  (1)
                                    #Occlusion check
                                    if occlusion_mask[y,x] == 0:#Not occluded point  
                                        left[y+yw,x+xw,j] = (rvalue * blending + left[y+yw,x+xw,j] * (1-blending))
                                        if splatting:
                                            right[y+yw,xd0+xw,j] = (((rvalue * blending + right[y+yw,xd0+xw,j] * (1-blending)) * (1-d1_blending)) + right[y+yw,xd0+xw,j] * d1_blending)
                                            if 0 <= xd1+xw and xd1+xw <= width-1:# Linear interpolation only if inside the border
                                                right[y+yw,xd1+xw,j] = (((rvalue * blending + right[y+yw,xd1+xw,j] * (1-blending)) * d1_blending) + right[y+yw,xd1+xw,j] * (1-d1_blending))
                                        else:
                                            right[y+yw,xd+xw,j] = rvalue * blending + right[y+yw,xd+xw,j] * (1-blending)
                                    elif not discard_occ:# Occluded point: Foreground point should be projected before occluded point
                                        if splatting:
                                            right[y+yw,xd0+xw,j] = (((rvalue * blending_occ + right[y+yw,xd0+xw,j] * (1-blending_occ)) * (1-d1_blending)) + right[y+yw,xd0+xw,j] * d1_blending)
                                            if 0 <= xd1+xw and xd1+xw <= width-1:
                                                right[y+yw,xd1+xw,j] = (((rvalue * blending_occ + right[y+yw,xd1+xw,j] * (1-blending_occ)) * d1_blending) + right[y+yw,xd1+xw,j] * (1-d1_blending))
                                            left[y+yw,x+xw,j] = ((right[y+yw,xd0+xw,j]*(1-d1_blending)+right[y+yw,xd1+xw,j]*d1_blending) * blending + left[y+yw,x+xw,j] * (1-blending))             
                                        else:
                                            right[y+yw,xd+xw,j] = rvalue * blending_occ + right[y+yw,xd+xw,j] * (1-blending_occ)
                                            left[y+yw,x+xw,j] = right[y+yw,xd+xw,j] * blending + left[y+yw,x+xw,j] * (1-blending)
                                            
                                else:#Left side occlusion (known) (2)
                                    left[y+yw,x+xw,j] = (rvalue * blending + left[y+yw,x+xw,j] * (1-blending))        
                                                    
                sample_i +=1

            x = x-1 if direction == 0 else x+1
    
    return sample_i

@nb.njit
def _bilateral_filling(dmap, img, n, o_xy = 2, o_i= 1, th=.001):
    h,w = img.shape[:2]
    assert dmap.shape == img.shape
    cmap = np.zeros_like(dmap)
    aug_dmap = dmap.copy()

    for y in range(h):
        for x in range(w):
            i_ref = img[y,x]
            d_ref = dmap[y,x]
            if d_ref > 0:
                for yw in range(-n,n+1):
                    for xw in range(-n,n+1):
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            weight = math.exp(-(((yw)**2+(xw)**2)/(2*(o_xy**2)) + ((img[y+yw,x+xw]-i_ref)**2)/(2*(o_i**2))))
                            if cmap[y+yw,x+xw] < weight:
                                cmap[y+yw,x+xw] = weight
                                aug_dmap[y+yw,x+xw] = d_ref
    
    aug_dmap = np.where(cmap>th,aug_dmap,0)

    return aug_dmap

def vsh(left, right, gt, occlusion_mask, guide_args):
    """
    
    """

    wsize = guide_args['vsh_patch_size']
    uniform_color = guide_args['vsh_uniform_patch']
    splatting = guide_args['vsh_splatting']
    method = guide_args['vsh_method']
    left2right = True
    o_xy = 2
    o_i= 1
    fillingThreshold = 0.001
    useFilling = guide_args['vsh_filling']
    blending_occ = 0.00
    discard_occ = False
    blending = guide_args['vsh_blending']
    stacking_method = guide_args['stacking_method']
    
    lc,rc = np.copy(left).astype(np.float32), np.copy(right).astype(np.float32)
    gt = gt.astype(np.float32)

    if stacking_method == 'VoxelGridEventStacking': 
        l_minvalue, l_maxvalue = np.percentile(lc[lc!=0], 5), np.percentile(lc[lc!=0], 95)
        r_minvalue, r_maxvalue = np.percentile(rc[rc!=0], 5), np.percentile(rc[rc!=0], 95)
        minvalue = min(l_minvalue, r_minvalue)
        maxvalue = max(l_maxvalue, r_maxvalue)
    else:
        l_minvalue, l_maxvalue = np.min(lc), np.max(lc)
        r_minvalue, r_maxvalue = np.min(rc), np.max(rc)
        minvalue = min(l_minvalue, r_minvalue)
        maxvalue = max(l_maxvalue, r_maxvalue)

    assert method in ["rnd",]
    direction = 1 if left2right else 0

    if len(lc.shape) < 3:
        lc,rc = np.expand_dims(lc, axis=-1), np.expand_dims(rc, axis=-1)
    
    #Convert rgb to "gray" if needed
    if len(lc.shape) >= 2 and lc.shape[2] >= 2:
        gray_context = np.mean(lc, axis=1)
    else:
        gray_context = np.squeeze(lc)

    if useFilling:
        filled_gt = _bilateral_filling(gt, gray_context, (wsize-1)//2, o_xy, o_i, th=fillingThreshold)
    else:
        filled_gt = gt

    if occlusion_mask is None:
        occlusion_mask = np.zeros_like(gt)        

    if method == "rnd":
        _virtual_projection_scan_rnd(lc,rc,filled_gt,uniform_color,wsize,direction, blending, blending_occ, occlusion_mask, discard_occ,splatting,minvalue,maxvalue)
    
    return lc,rc
