import re
import numpy as np
import scipy.ndimage as nd
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import UnivariateSpline


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where 
                    data value shoud be replaced.
                    If None (defaut), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 

    Modified From:
        https://stackoverflow.com/questions/3662361/fill-in-missing-values-
                    with-nearest-neighbour-in-python-numpy-masked-arrays
    """
    if invalid is None:
        invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def gridize(x, y, data):
    '''
    Take spatially-referenced column data and turn it into 2D array
    '''
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    new = np.empty(x_vals.shape + y_vals.shape)
    new.fill(np.nan)  # or whatever your desired missing data flag is
    new[x_idx, y_idx] = data
    return new


def degridize(data):
    '''
    Take 2D array and then turn it into spatially-referenced column data
    '''
    x_size, y_size = data.shape[1], data.shape[0]
    columns = [(x, y, data[y, x]) for x in range(x_size)
               for y in range(y_size)]
    return np.array(columns)


def find_below(surface, points):
    '''
    Use surface data (in pandas DataFrame format) to determine if points
        (n by 2 array format) are below surface

    Returns boolean index array for masking PIVLab column data
    '''
    surface = surface.dropna()
    sf = UnivariateSpline(surface.x, surface.y)
    x, y = points[:, 0], points[:, 1]
    new_y = y - sf(x)
    return new_y < 0


def roi_geom(filename):
    '''
    Reads a PIVlab output file and extracts information about the  region
        of interest, or the selected observation window, outside of which 
        data is ignored

    Parameters
    ----------
    filename : str
        Name/location of PIV file containing x,y,u,v data.

    Returns
    -------
    pd.DataFrame
        Compilation of measurements of the selected PIV region of interest.

    '''
    # read data
    data = pd.read_csv(filename, skiprows=3, usecols=[0, 1, 2, 3],
                       names=['x', 'y', 'u', 'v'])
    width = data.x.max() - data.x.min()
    height = data.y.max() - data.y.min()
    # return width,height,xmin,xmax,ymin,ymax
    bound = [data.x.min(), data.x.max(), data.y.min(), data.y.max()]
    bound = np.array(bound).reshape((1, len(bound)))
    bound = pd.DataFrame(bound, columns=['xmin', 'xmax', 'ymin', 'ymax'])
    return width, height, bound


def PIV_framenumbers(filename):
    '''
    Read A/B frame numbers from PIVLab output.
    Assumes that the frame number is the last number in the filename before
        extension.
    Also reads filename for input, assuming filenumber is last number before 
        extension.
    '''
    frame_line = []
    f = open(filename)
    lines = f.readlines()
    frame_line = lines[1]
    filenumber = re.findall('\d+', filename)[-1]
    A_frame = re.findall('\d+', frame_line.split()[4])[-1]
    B_frame = re.findall('\d+', frame_line.split()[7])[-1]
    f.close()
    return int(filenumber), int(A_frame), int(B_frame)


def do_kdtree(target_points, orig_points, find_num):
    '''
    Simple wraparound to compute k-d Tree for k-dimensional data

    Returns
    -------
    dist
        -Euclidean distance between points
    indicies
        -indicies of matched closest points
    '''
    mytree = KDTree(target_points, balanced_tree=False, compact_nodes=False)
    dist, indicies = mytree.query(list(orig_points), k=find_num, workers=-1)
    return dist, indicies
