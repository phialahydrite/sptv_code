#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:37:33 2022

@author: Phiala Thouvenin
"""
from scipy import ndimage as nd
from scipy import ndimage
from scipy.ndimage import filters
from scipy.spatial import distance
import glob, re
import pims
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
from scipy.interpolate import UnivariateSpline
from numpy.linalg import lstsq
import time
from os.path import exists
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

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

def gridize(x,y,data):
    '''
    Take spatially-referenced column data and turn it into 2D array
    '''
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    new = np.empty(x_vals.shape + y_vals.shape)
    new.fill(np.nan) # or whatever your desired missing data flag is
    new[x_idx, y_idx] = data
    return new

def degridize(data):
    '''
    Take 2D array and then turn it into spatially-referenced column data
    '''
    x_size, y_size = data.shape[1], data.shape[0]
    columns=[(x,y,data[y,x]) for x in range(x_size) for y in range(y_size)]
    return np.array(columns)


def particle_displacer(piv_files,
                        ero_temps,front,
                        particles,
                        framenumbers,
                        surface_hdf5,
                        begin_file,end_file,
                        step = 2, radius=24, 
                        verbose = False,
                        replace_nan = False,
                        fill_array = False,
                        mask_piv = False):
    '''                       
    Using Matlab PIvab output with x,y,u,v data [files], displace synthetic 
        markers [particles], while simutaneously filtering PIV data that 
        falls outside of the model wedge, using calcuated surfaces 
        [surface_hdf5], between a range of files [begin_file:end_file]
    '''
    #initial variables and lists
    start_frames = particles.frame.unique()
    df = [] #DataFrame
    rolling_particle_nos = 0 # inital count of particles to add to
    total_time = float()
    for n in start_frames:
        # unique particles for each frame in constructed grid of particles
        parts = particles[particles.frame == n]
        # first compared B-image for dataframe
        B_frame_start = framenumbers[framenumbers.B_frame >= n].index.min()
        continuing = []
        for i, f in enumerate(piv_files[B_frame_start:end_file]):
            start_time=time.time()
            # read frame numbers from PIV output
            fn, A_img_no, B_img_no = PIV_framenumbers(f)
            
            # read PIV data, ignoring extra columns if they exist
            data = pd.read_csv(f,skiprows=3,usecols = [0,1,2,3],\
                                names=['x','y','u','v'])
            
            # fill nan with zeros, do not use if using fill_array option
            if replace_nan:
                data.loc[np.isnan(data.u)].u = 0.
                data.loc[np.isnan(data.v)].v = 0.
            
            # calcuate 2d mapts of displacement values
            U = gridize(data.x,data.y,data.u)
            V = gridize(data.x,data.y,data.v)   
            
            # fill array nans with nearest valid neighbors
            if fill_array:
                U = fill(U)
                V = fill(V)
            
            # calculate shear strain         
            dUdY = np.gradient(U.T)[0] 
            dVdX = np.gradient(V.T)[1]
            ep_xy = 0.5 * (dUdY + dVdX)
            data['exy'] = degridize(ep_xy)[:,2] 
            
            # calcuate vorticity (Meynart, Fung, Eringen, Timoshekno, etc...)
            VO = dVdX - dUdY
            data['vort'] = degridize(VO)[:,2] 

            # flip PIVlab format upside-down and shift on x-axis
            roiwidth, roiheight, bound = roi_geom(pfiles[0])
            roixmin = bound.xmin.values
            roiymax = bound.ymax.values
            data.v = -data.v
            data.y = roiymax-data.y
            data.x = data.x - roixmin   
            # data.y = data.y.max()-data.y
            # data.x = data.x - x_min
            
            # read  and merge temperature data, shifting coordinates
            temp_e = pd.read_csv(ero_temps[i], delimiter=' ',
                                names=['x','y','temp_c'])
            
            # place temps in data by merging to align with existing values
            data = data.merge(temp_e,left_on=['x','y'],right_on=['x','y'],
                                how='left')
            # this is not needed if already masked in PIVlab
            if mask_piv: 
                xy = np.array([data.x,data.y]).T
                # read surface
                surface = pd.read_hdf(surface_hdf5,'wedgetop_%05.0f'%B_img_no)
                surface.x = surface.x-xmin
                surface.y = surface.y-(images.frame_shape[0] - ymax)
                
                # return boolean index of points beneath surface
                subsurface_piv = find_below(surface,xy)
        
                # remove background velocities 
                data = data.loc[subsurface_piv]
            # background displacement coordinate values for kd-tree
            xy_target = np.vstack((data.x,data.y)).T
            
            # INITIALIZE particle displacement arrays    
            if i == 0: 
                # initial particle location grid for kd-tree        
                xy_orig = parts[['x','y']].values
                
                # perform kd-tree, only using nearest PIV grid point
                dist,ind = do_kdtree(xy_target,xy_orig,1)
                
                # find associated displacements, strains, and vorticity 
                #   within radius
                u,v,exy,vort,e_temp = np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig)):
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig)):
                    if close_ind[k] != None: #ignore empty                
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                        e_temp[k] = data.temp_c.iat[close_ind[k]]
                nxs = parts.x.values + u
                nys = parts.y.values + v
                nexy = exy
                nvort = vort
                ne_temp = e_temp
                
                # shifted x-value
                f_data = front[front.frame == B_img_no]
                if len(f_data) > 0:
                    f_pos = f_data.x_df.values
                else:
                    f_pos = front[front.frame == A_img_no].x_df.values
                nxs_dfm = nxs - f_pos
                
                first_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                first_frame = np.ones(len(first_pnums))*B_img_no
                first = pd.DataFrame(np.vstack((first_frame,first_pnums,
                                        nxs,nys,nexy,nvort,ne_temp,nxs_dfm)).T,\
                                        columns = ['frame','particle','x','y',
                                                    'exy','vort','e_temp',
                                                    'x_df'])
                    
            # CALCULATE incremental displacement arrays for each particle    
            else:
                # updated particle location grid for kd-tree
                xy_orig_update = np.array([nxs,nys]).T
                
                # updated kd-tree for new particle locations, using one point
                dist,ind = do_kdtree(xy_target,xy_orig_update,1)
                
                # calcuate displacements, strains, and vorticity within radius
                u,v,exy,vort,e_temp = np.zeros(len(xy_orig_update)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig)),\
                                            np.zeros(len(xy_orig))
                close_ind =[]
                for k in range(len(xy_orig_update)):
                    #determine location of nearest grid points
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts)>=1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig_update)):
                    if close_ind[k] != None: #ignore empty                
                        u[k] = data.u.iat[close_ind[k]]
                        v[k] = data.v.iat[close_ind[k]]
                        exy[k] = data.exy.iat[close_ind[k]]
                        vort[k] = data.vort.iat[close_ind[k]]
                        e_temp[k] = data.temp_c.iat[close_ind[k]]
                nxs = nxs + u
                nys = nys + v
                nexy = exy
                nvort = vort
                ne_temp = e_temp
                
                # shifted x-value
                f_data = front[front.frame == B_img_no]
                if len(f_data) > 0:
                    f_pos = f_data.x_df.values
                else:
                    f_pos = front[front.frame == front.frame.max()].x_df.values
                nxs_dfm = nxs - f_pos
                
                # compile further displaced particle and frame information
                continuing_pnums = np.arange(len(parts.particle.unique())) + \
                                        rolling_particle_nos
                continuing_frame = np.ones(len(continuing_pnums))*B_img_no
                continuing_df = pd.DataFrame(np.vstack((continuing_frame,\
                                    continuing_pnums,nxs,nys,nexy,nvort,
                                    ne_temp,nxs_dfm)).T,\
                                    columns = ['frame','particle','x','y',
                                                'exy','vort','e_temp','x_df'])
                continuing.append(continuing_df)
            if verbose:
                frame_elapsed = time.time()-start_time
                total_time += frame_elapsed
                num_displaced = len([i for i in close_ind if i is not None])
                print('Calcuated displacements for %g '%(num_displaced) + \
                        'particles for File %05.0f, '%(fn) + \
                        'starting at Frame %05i, '%(n) + \
                        'Frames %05.0f and %05.0f '%(A_img_no,B_img_no) + \
                        'in %06.3f seconds, '%(frame_elapsed) + \
                        '%010.3f total seconds elapsed.'%(total_time))
                        
        # compile data (may be large, >20GB)
        # keep NaNs to filter later, if needed
        df.append(pd.concat((first,pd.concat(continuing))))
        # count of particles as they are iteratively added
        rolling_particle_nos += len(parts.particle.unique())
    return pd.concat(df).reset_index(drop=True)

def deformation_front_mode(surfs,num_frames,far_edge_count=1000,threshold=10,
                            first_frames=1000,horiz_width=2000,
                            from_larger=True, grow_horiz=True,
                            vergence='s'):
    '''     
    Examine calcuated surfaces, and determine the surface expression of their
        deformation fronts over the length of an experiment.
    Perform calcuation on surface with the retrowedge as the first area that 
        exceeds the average flat topography elevation, and the deformation 
        front defined as the last area that exceeds elevation outside of 
        a specifed range of [sand_elev +/- surfvar].
    Calcuated value is only the physical expression of the deformation front, 
        as actual deformation may jump forelandward for a short time 
        before surface expression.         
    '''
    # determine maxlimum width of profile
    len_x = []
    for i in range(num_frames):
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        if from_larger:
            surf.x = surf.x - x_min
            surf.y = surf.y - (im_h-ymax)
        len_x.append(surf.x.max())
    width = int(max(len_x))    
    
    # place surface in preallocated array
    surface = np.zeros((num_frames,width))
    for i in range(num_frames):        
        surf = pd.read_hdf(surfs,'wedgetop_%05.0f'%(i))
        surf = surf.y.dropna().reindex(surf.x, method='nearest').reset_index()
        if from_larger:
            surf.x = surf.x - x_min
            surf.y = surf.y - (im_h-ymax)
        surface[i,np.array(surf.x.values,dtype='int')-1] = surf.y.values
        
    # determine mean height of foreland material on far edge of surface array
    far_edge_mean = surface[:first_frames,-far_edge_count:].mean()
    # mask to find area that is within a given elevation of this mean
    mask = (surface > far_edge_mean - threshold) & \
                (surface < far_edge_mean + threshold)
    # set boolean mask on edges to remove edge effects
    if grow_horiz:
        mask = np.hstack((mask,np.ones((mask.shape[0],horiz_width)))).astype(bool)
    mask[:,-far_edge_count:] = True
    
    # different modes for either style of model, singly or doubly vergent
    if vergence == 's':
        edge = filters.sobel(mask.astype(float))
        front = []
        for i in range(edge.shape[0]):
            if len(edge[i,:][edge[i,:] > 0]):
                front_loc = np.nonzero(edge[i,:])[0][-1]
                front.append([i,front_loc])
            else:
                front.append([i,np.nan])
        front = pd.DataFrame(front,columns=('frame','x_df'))
    if vergence == 'd':
        edge = filters.sobel(mask.astype(float))
        front = []
        for i in range(edge.shape[0]):
            if len(edge[i,:][edge[i,:] > 0]):
                retro_loc = np.nonzero(edge[i,:])[0][0]
                front_loc = np.nonzero(edge[i,:])[0][-1]
                topo_div = round(np.where(surface[i,:] == surface[i,:].max())[0].mean())
                front.append([i,retro_loc,front_loc,topo_div])
            else:
                front.append([i,np.nan,np.nan])
        front = pd.DataFrame(front,columns=('frame','x_rf','x_df','x_td'))
    elif vergence != 's' and vergence != 'd':
        raise ValueError('Choose either singly (s) or doubly (d) vergent mode') 
    return front

def find_below(surface,points):
    '''
    Use surface data (in pandas DataFrame format) to determine if points
        (n by 2 array format) are below surface
    
    Returns boolean index array for masking PIVLab column data
    '''
    surface = surface.dropna()
    sf = UnivariateSpline(surface.x, surface.y)
    x, y = points[:,0], points[:,1]
    new_y = y - sf(x)
    return new_y<0

def PIV_framenumbers(filename):
    '''
    Read A/B frame numbers from PIVLab output.
    Assumes that the frame number is the last number in the filename before
        extension.
    Also reads filename for input, assuming filenumber is last number before 
        extension.
    '''
    frame_line = [] 
    f=open(filename)
    lines=f.readlines()
    frame_line = lines[1]
    filenumber = re.findall('\d+', filename)[-1]
    A_frame = re.findall('\d+',frame_line.split()[4])[-1]
    B_frame = re.findall('\d+',frame_line.split()[7])[-1]
    f.close()
    return int(filenumber), int(A_frame), int(B_frame) 

def tp_plot_traj(trajs,sample_int=1,particle_int=1,scaled=False,save=False,
                cmap=plt.cm.viridis):
    '''
    Convenience function to plot all trajs in a given dataframe
        with scaled axes.
    Optional settings for plotting every [sample_int] particle location and/or
        every [particle_int] particle, useful for extremely large datasets.
    '''
    f, ax = plt.subplots(figsize=(10,10))
    particle_it = np.sort(trajs['particle'].unique())
    trajs = trajs[trajs.particle.isin(particle_it)]
    # initialize plot 
    #   place ticks outside of plot to avoid covering image
    #   remove right and upper axes to simplify plot
    #   only plot ticks on the left and bottom of plot
    ax.tick_params(axis='y', direction='out')
    ax.tick_params(axis='x', direction='out')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot particle trajs leading up to current frame
    if scaled:
        x = trajs['x']/scale
        y = trajs['y']/scale
    else:
        x = trajs['x']
        y = trajs['y']
    ax.scatter(x[::sample_int],y[::sample_int],
                c=trajs['frame'][::sample_int],marker='o',
                s=2, alpha=0.25, cmap=cmap,lw=0,
                vmin=trajs.frame.min(),vmax=trajs.frame.max())
    plt.axis('scaled')
    if scaled:
        ax.set_xlim([0,im_w/scale])
    plt.show()
        
def depth_synthetic_hist(tp_df,im_w,im_h,num_frames,square_size=8):
    '''
    Calcuate a histogram of depth for large collections of synthetic particles
    '''
    # Set divide-by-zero warning to off to ignore warning.
    np.seterr(divide='ignore', invalid='ignore')
    # set the range of histograms, number of frames by image (y,x) dimensions
    extent = [[0,num_frames],[0,im_h],[0,im_w]]
    # calcuate the correct number of bins in each spatial dimension to match
    #   ideal square size
    r_im_h = round(im_h/square_size)*square_size
    r_im_w = round(im_w/square_size)*square_size
    num_x = len(np.linspace(0,r_im_w,int(round(im_w/square_size))))
    num_y = len(np.linspace(0,r_im_h,int(round(im_h/square_size))))
    # calcuate burial, exhumation, and minimum depth values
    #   additionally calcuate counts within each bin to get mean 
    #   values of each quantity h
    particle_pos = tp_df[['frame','y','x']].values
    depth, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x), 
                                weights=tp_df.depth, 
                                normed=False, range=extent)
    starting_depth, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x), 
                                        weights=tp_df.starting_depth, 
                                        normed=False, range=extent)
    max_depth, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x), 
                                    weights=tp_df.max_depth, 
                                    normed=False, range=extent)
    cum_max_depth, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x), 
                                        weights=tp_df.cum_max_depth, 
                                        normed=False, range=extent)
    geot_e, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x), 
                                weights=tp_df.e_temp, 
                                normed=False, range=extent)
    geot_b, _ = np.histogramdd(particle_pos, bins=(num_frames,num_y,num_x), 
                                weights=tp_df.b_temp, 
                                normed=False, range=extent)
    part_counts, bin_edges, = np.histogramdd(particle_pos,
                                            bins=(num_frames,num_y,num_x),
                                            range=extent)
    # if nan, mask array for cleanliness of data
    depth = np.ma.masked_invalid(depth / part_counts)
    starting_depth = np.ma.masked_invalid(starting_depth / part_counts)
    max_depth = np.ma.masked_invalid(max_depth / part_counts)
    cum_max_depth = np.ma.masked_invalid(cum_max_depth / part_counts)
    geot_e = np.ma.masked_invalid(geot_e / part_counts)
    geot_b = np.ma.masked_invalid(geot_b / part_counts)
    return depth,starting_depth,max_depth,cum_max_depth,geot_e,geot_b
    
def first_closest(sample, pivot, k):
    '''
    Find k-nearest points from pivot in sample, pick first-ocurring instance
    '''
    nearest = sorted(enumerate(sample), key=lambda nv: abs(nv[1] - pivot))[:k]
    return min([n[0] for n in nearest])

def starting_max_depth_redux(df):
    '''
    Caclulate the starting depth of each grouped particle into a flattened 
        list to then merge with DataFrame
    Returns starting depth, max depth, frame where max depth was reached
    '''
    grouped_df = df.groupby('particle')
    sd,md,mdf = [],[],[]
    for i in df.particle.unique():
        group = grouped_df.get_group(i)
        sd.append(group.depth.head(1).values * np.ones(len(group)))
        max_depth=group.depth.min()
        try:
            max_depth_frame=group[group.depth==max_depth].frame.values[0]
        except:
            max_depth_frame=np.nan
        md.append(max_depth * np.ones(len(group)))
        mdf.append(max_depth_frame * np.ones(len(group)))
    sd = np.concatenate(sd).ravel().tolist()
    md = np.concatenate(md).ravel().tolist()
    mdf = np.concatenate(mdf).ravel().tolist()
    return sd, md, mdf

def depth_calc(img_h,trajs,frame_spacing,surfname,preflipped=False,silent=True,
                shift_surface=False,from_larger=False,skip_edge=True,
                x_min=0,y_max=0,edge_buf=20):
    '''
    USE THIS:
    This version of the function caluates depth for each particle at each
        frame en masse, rather than calcuate an interpolation for each particle
        individually at each frame. This is roughly 1000X faster than other 
        functions
    Modified for PIV pseudoparticle tracking, with simpler column format
    Modified to accept surfaces in uncropped image coordinates
    Modified to add strain components
    Modified to add temperature pass-through option
    ----------
    Parameters
    ----------
    img_h: float
        height of imput image
    trajs : trackpy trajectories
        Linked particle paths calcuated from trackpy
    frame_spacing : integer
        'distance' between frames that change in depth shoud be calcuated 
    surfname : HDF5 store
        contains the model topography of every image present in experiment
        
    Returns
    -------
    df : Pandas DataFrame
        a DataFrame containing all corresponding data from trajs, with addition
        of depth other columns. 
    '''
    # find image height for shifting
    # unique elements of frames to loop over 
    unique_frames = trajs.frame.unique()[::frame_spacing]
    # store interpolation functions for entire experiment in list
    #   this is to avoid interpolation being done repeatedly unnecessarily
    s_funcs=[]
    for fr in unique_frames:
        # corrsponding section of trajs for each combo 

        # read in calcuated surface for depth calcuation
        try:
            su = pd.read_hdf(surfname,'wedgetop_%05.0f'%fr).dropna(how='any')
        except:
            su = pd.read_hdf(surfname,'wedgetop_%05.0f'%0).dropna(how='any')
        # if back-of-box edge of surface has edge effects, only use surface
        #   past [edge_buf] pixels
        if skip_edge:
            su = su[edge_buf:]        
        # remove any NaNs
        su.dropna(inplace=True)

        # if surfaces were calcuated on larger image and left in those coord.,
        #   shift them so that the surface starts at x=0,y=surface elevation
        #   from base
        if from_larger:
            su.x = su.x - x_min
            su.y = su.y - (img_h-y_max)
        # create univariate interpolation function for surface
        #   (some trajectories fall between pixels)
        if preflipped:
            sf = UnivariateSpline(su.x, su.y)
        else:
            sf = UnivariateSpline(su.x, img_h - su.y)
        s_funcs.append((fr,sf))
        if not silent:
            print('Generated topography function for Frame %g'%fr)
    # pu just the frame # for future use
    f_idx = [s_funcs[i][0] for i in range(len(s_funcs))]
    # loop over all possible combinations
    df = []
    for fr in unique_frames:
        sel = trajs[(trajs.frame == fr)].sort_values(by=['particle'])
        if not sel.empty:            
            # calcuate depth for every point (from stored surface functions)
            depth = sel.y - s_funcs[f_idx.index(fr)][1](sel.x)
            if not silent:
                print('Calcuated Depth for %g Particles at Frame %g'%(len(depth),fr))
            # popuate DataFrame to mimic the format of trackpy DataFrames
            sel_dat = sel.join(depth.rename('depth')) # y -> depth rename
            df.append(sel_dat)
    return pd.concat(df).sort_values(by=['particle','frame']).reset_index(drop=True)

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
    data = pd.read_csv(filename,skiprows=3,usecols = [0,1,2,3],\
                        names=['x','y','u','v'])
    width = data.x.max() - data.x.min()
    height  = data.y.max() - data.y.min()
    #return width,height,xmin,xmax,ymin,ymax
    bound = [data.x.min(),data.x.max(),data.y.min(),data.y.max()]
    bound = np.array(bound).reshape((1,len(bound)))
    bound = pd.DataFrame(bound,columns=['xmin','xmax','ymin','ymax'])
    return width,height,bound

def do_kdtree(target_points,orig_points,find_num):
    '''
    Simple wraparound to compute k-d Tree for k-dimensional data
    
    Returns
    -------
    dist
        -Euclidean distance between points
    indicies
        -indicies of matched closest points
    '''
    mytree = KDTree(target_points,balanced_tree=False,compact_nodes=False) 
    dist, indicies = mytree.query(list(orig_points),k=find_num,workers=-1)
    return dist, indicies

'''
Generate particles for one experiment.
Define basic parameters and generate synthetic particle grid
'''
#pjt, high friction, 15deg prebuilt taper, 9ero,glassbeads 1cm 
x_min,y_max = 140, 1220 
x_max,y_min = x_min+4400, 400    
prefix = 'pjt_highfric_15deg_1cmglassbeads_9ero'
images = pims.ImageSequence('pjt_highfric_15deg_1cmglassbeads_9ero/*.jpg')
h5name = 'pjt_highfric_15deg_1cmglassbeads_9ero/pjt_highfric_15deg_1cmglassbeads_9ero_trajs_size11.h5'
surfs = 'pjt_highfric_15deg_1cmglassbeads_9ero/pjt_highfric_15deg_1cmglassbeads_9ero_surfnocv.h5'
xmin,xmax,ymin,ymax = 140,x_min+4400,400,1220
pixelmm_scale = 3.05
scale=61.
pfiles = glob.glob('pjt_highfric_15deg_1cmglassbeads_9ero/piv_nomask/*.txt')
pfiles.sort()
fileinfo = [np.array(PIV_framenumbers(f)) for f in pfiles]
fileinfo = pd.DataFrame(fileinfo,columns=['filenumber','A_frame','B_frame'])
# ero_events = pd.read_excel('pjt_highfric_15deg_1cmglassbeads_9ero/pjt_highfric_15deg_1cmglassbeads_9ero_erosional_events.xsx')
im_w=images.frame_shape[1]
im_h=images.frame_shape[0]
end_cutoff = len(images)
verge = 'single'
###start and skip, 2 anf 4 or most models
frame_start = 2
frame_skip=4
#erosional temperatures as calculated using erosional_thickness_and_time
efiles = glob.glob('pjt_highfric_15deg_1cmglassbeads_9ero_118kappa_0csurf_temps_cumsum_wh/ero_temps_*.txt')
efiles.sort()
chan_plot = False
exh = True
print('Processing %s'%prefix)

'''
PARTICLE PIV DISPLACEMENT AND GRID GENERATION
'''
# get reigon-of-interest information
roiwidth, roiheight, _ = roi_geom(pfiles[0])

# spacing of particles and frames before adding new particles
part_spacing = 4
frame_spacing = scale
extra = 4
step=2 # PIV analysis step
begin_file = 0
end_file = int(end_cutoff / step)
begin_frame = begin_file * step
end_frame = end_file * step

## read surface
surface = pd.read_hdf(surfs,'wedgetop_%05.0f'%begin_frame)
surface.x = surface.x - xmin
surface.y = surface.y - (im_h-ymax)

## initial grid covering entire area within region of interest
Xs = np.linspace(0,roiwidth,round(roiwidth/part_spacing))
Ys = np.linspace(0,roiheight,round(roiheight/part_spacing))
grid = [(x,y) for x in Xs for y in Ys]
grid.sort()
grid = pd.DataFrame(np.array(grid),columns=['x','y'])

## return boolean index of points beneath surface
subsurface = find_below(surface,grid.values)
grid = grid[subsurface]
grid['particle'] = list(range(len(grid)))
grid['frame'] = 0

## grid that will be repetitively added each [frame_spacing*extra] frames
## 	this is at the far end of the model where a gap opens with lateral 
##	motion of the model
box_w = roiwidth-(pixelmm_scale*frame_spacing*extra)
box_h = roiheight
Xs = np.linspace(box_w,roiwidth,round((pixelmm_scale*frame_spacing)/part_spacing))
Ys = np.linspace(0,box_h,round(box_h/part_spacing))
grid_sel = [(x,y) for x in Xs for y in Ys]
grid_sel.sort()
grid_sel = pd.DataFrame(np.array(grid_sel),columns=['x','y'])
grid_sel = grid_sel[grid_sel.y <= 2*scale]
grid_sel['particle'] = np.array(list(range(len(grid_sel)))) + len(grid)
grid_sel['frame'] = 0

## grid on bottom of box that is repetitively added so to not leave gaps
##     will be added each [frame_spacing] frames
box_w = 0 # all the way to other end
box_h = 2*scale # in centimeters
Xs = np.linspace(box_w,roiwidth,round((pixelmm_scale*frame_spacing)/part_spacing))
Ys = np.linspace(0,box_h,round(box_h/part_spacing))
grid_sel_bottom = [(x,y) for x in Xs for y in Ys]
grid_sel_bottom.sort()
grid_sel_bottom = pd.DataFrame(np.array(grid_sel_bottom),columns=['x','y'])
grid_sel_bottom['particle'] = np.array(list(range(len(grid_sel_bottom)))) \
                                        + len(grid) + len(grid_sel)
grid_sel_bottom['frame'] = 0

## add this bottom layer of particles if desired
# later_grids = pd.concat((grid_sel,grid_sel_bottom))
later_grids = grid_sel


## compilation of original grids and these repetitively added grids
added_grids = []
for i in list(range(int(end_frame/frame_spacing)))[1:-1]:
    # further grids   
    grid_d = later_grids.copy()
    grid_d.particle = np.array(list(range(len(grid_d)))) + i*len(grid_sel) \
                                        + i*len(grid_sel_bottom)
    grid_d.frame = i * frame_spacing
    added_grids.append(grid_d)
all_grids = pd.concat((grid,pd.concat(added_grids))).reset_index()
#all_grids = pd.concat(added_grids)

## calcuate particle displacements
# pts_fname = '%s_artificial_pts_addbottomline_temp.csv'%prefix
pts_fname = '%s_artificial_pts_temp.csv'%prefix
## x-values shifted to location of deformation front
if verge == 'single':
    fronts = deformation_front_mode(surfs,end_frame,from_larger=True,
                                    vergence='s')
else:
    fronts = deformation_front_mode(surfs,end_frame,from_larger=True,
                                    vergence='d')
pts = particle_displacer(pfiles,efiles,fronts,
                        all_grids,fileinfo,surfs,0,
                        end_file,verbose=False,fill_array=True,
                        radius=24)
# cumulative values
pts['cumexy'] = pts.groupby('particle')['exy'].cumsum()  
pts['cumvort'] = pts.groupby('particle')['vort'].cumsum() 
pts['cum_conv'] = pts.frame * pixelmm_scale
pts.to_csv(pts_fname,chunksize=1000000,float_format='%.8f',index=False)
del pts