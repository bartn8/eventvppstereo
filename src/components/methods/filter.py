import numpy as np
from numba import njit


@njit
def _left_warp(dmap):
    """
    Warp left disparity map to right view.
    Original values are preserved.
    Interpolation is not applied, only round.
    Uniqueness check: if a point collide then save max value.
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map in the left view    
    
    Returns
    -------
    omap: HxW np.ndarray
        Original disparity map warped to right view (occlusion map)
    """
    h,w = dmap.shape[:2]
    omap = np.zeros(dmap.shape, dtype=dmap.dtype)

    #Warp left dmap in occlusion dmap
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                d = int(round(dmap[y,x]))
                xd = x-d
                if 0 <= xd and xd <= w-1:
                    if omap[y,xd] < dmap[y,x]:
                        omap[y,xd] = dmap[y,x]
    
    return omap

@njit
def _weighted_conf(dmap, rx=9, ry=7, l=2, g=0.4375, th=1.1):
    """
    Return a confidence map based on weighted distance.
    Points that are too close to foreground pixel are rejected (conf=1)
    
    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map used to extract confidence map.
    rx: int
        Horizontal search radius (1,3,5,...)
    ry: int
        Vertical search radius (1,3,5,...)
    th: float
        Threshold for absolute difference

    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """      

    h,w = dmap.shape[:2]
        
    #Confidence map between 0 and 1 (binary)
    conf_map = np.zeros(dmap.shape, dtype=np.uint8)

    rx = rx//2  
    ry = ry//2    

    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                for xw in range(-rx,rx+1):               
                    for yw in range(-ry-1,ry+1):                     
                        if 0 <= y+yw and y+yw <= h-1 and 0 <= x+xw and x+xw <= w-1:
                            if dmap[y+yw, x+xw] > 0:
                                #Check that's a "background point"
                                #For slanted surfaces: check later with a threshold
                                if dmap[y+yw, x+xw] < dmap[y,x]:
                                    #Use Manhattan distance to keep in mind y-shifts
                                    #Reject a point if foreground disparity is greather than distance between fg and bg
                                    if (dmap[y,x]-dmap[y+yw, x+xw]) - l*(g*abs(xw)+(1-g)*abs(yw)) > th:
                                        conf_map[y+yw, x+xw] = 1                                            
                    
            else:
                conf_map[y,x] = 1
                    
    return conf_map

@njit
def _conf_unwarp(conf, dmap, do_flip = False):
    """
    Unwarp the confidence map to left view.
    Original values are preserved.
    Interpolation is not applied, only round.

    Parameters
    ----------
    conf: HxW np.ndarray
        Confidence map to unwarp.
    dmap: HxW np.ndarray
        Disparity map for warping operation.
    
    Returns
    -------
    conf_rst: HxW np.ndarray
        Unwarped confidence map
    """    
    h,w = dmap.shape[:2]
    conf_rst = np.ones(conf.shape, dtype=conf.dtype)

    #Warp occlusion dmap in left dmap
    for y in range(h):
        for x in range(w):
            if dmap[y,x] > 0:
                d = int(round(dmap[y,x]))
                xd = x+d if not do_flip else x-d
                if 0 <= xd and xd <= w-1:
                    conf_rst[y,xd] = conf[y,x]
    
    return conf_rst

def occlusion_heuristic(dmap, rx=9, ry=7, l=2, g=0.4375, th_conf=1):
    """
    Occlusion filter based on a weighted window.

    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.  
    rx: int 
        x-axis radius of the window
    ry: int 
        y-axis radius of the window
    th_conf: float
        confidence threshold: used to classify a occluded point
    Return
    ------
    conf_map: HxW np.ndarray
        Binary confidence map: 0 for no occlusion
    """

    omap = _left_warp(dmap)
    conf_map = _weighted_conf(omap,rx=rx, ry=ry, l=l, g=g, th=th_conf)  
    conf_map = _conf_unwarp(conf_map, omap)
        
    return conf_map