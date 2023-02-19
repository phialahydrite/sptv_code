import time
import pandas as pd
import numpy as np
from scipy.ndimage import filters
from main import pfiles, images, im_h, x_min, xmin, ymax
from util import *


def deformation_front_mode(surfs, num_frames, far_edge_count=1000, threshold=10,
                           first_frames=1000, horiz_width=2000,
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
        surf = pd.read_hdf(surfs, 'wedgetop_%05.0f' % (i))
        if from_larger:
            surf.x = surf.x - x_min
            surf.y = surf.y - (im_h-ymax)
        len_x.append(surf.x.max())
    width = int(max(len_x))

    # place surface in preallocated array
    surface = np.zeros((num_frames, width))
    for i in range(num_frames):
        surf = pd.read_hdf(surfs, 'wedgetop_%05.0f' % (i))
        surf = surf.y.dropna().reindex(surf.x, method='nearest').reset_index()
        if from_larger:
            surf.x = surf.x - x_min
            surf.y = surf.y - (im_h-ymax)
        surface[i, np.array(surf.x.values, dtype='int')-1] = surf.y.values

    # determine mean height of foreland material on far edge of surface array
    far_edge_mean = surface[:first_frames, -far_edge_count:].mean()
    # mask to find area that is within a given elevation of this mean
    mask = (surface > far_edge_mean - threshold) & \
        (surface < far_edge_mean + threshold)
    # set boolean mask on edges to remove edge effects
    if grow_horiz:
        mask = np.hstack(
            (mask, np.ones((mask.shape[0], horiz_width)))).astype(bool)
    mask[:, -far_edge_count:] = True

    # different modes for either style of model, singly or doubly vergent
    if vergence == 's':
        edge = filters.sobel(mask.astype(float))
        front = []
        for i in range(edge.shape[0]):
            if len(edge[i, :][edge[i, :] > 0]):
                front_loc = np.nonzero(edge[i, :])[0][-1]
                front.append([i, front_loc])
            else:
                front.append([i, np.nan])
        front = pd.DataFrame(front, columns=('frame', 'x_df'))
    if vergence == 'd':
        edge = filters.sobel(mask.astype(float))
        front = []
        for i in range(edge.shape[0]):
            if len(edge[i, :][edge[i, :] > 0]):
                retro_loc = np.nonzero(edge[i, :])[0][0]
                front_loc = np.nonzero(edge[i, :])[0][-1]
                topo_div = round(
                    np.where(surface[i, :] == surface[i, :].max())[0].mean())
                front.append([i, retro_loc, front_loc, topo_div])
            else:
                front.append([i, np.nan, np.nan])
        front = pd.DataFrame(front, columns=('frame', 'x_rf', 'x_df', 'x_td'))
    elif vergence != 's' and vergence != 'd':
        raise ValueError('Choose either singly (s) or doubly (d) vergent mode')
    return front


def particle_displacer(piv_files,
                       ero_temps, front,
                       particles,
                       framenumbers,
                       surface_hdf5,
                       begin_file, end_file,
                       step=2, radius=24,
                       verbose=False,
                       replace_nan=False,
                       fill_array=False,
                       mask_piv=False):
    '''                       
    Using Matlab PIvab output with x,y,u,v data [files], displace synthetic 
        markers [particles], while simutaneously filtering PIV data that 
        falls outside of the model wedge, using calcuated surfaces 
        [surface_hdf5], between a range of files [begin_file:end_file]
    '''
    # initial variables and lists
    start_frames = particles.frame.unique()
    df = []  # DataFrame
    rolling_particle_nos = 0  # inital count of particles to add to
    total_time = float()
    for n in start_frames:
        # unique particles for each frame in constructed grid of particles
        parts = particles[particles.frame == n]
        # first compared B-image for dataframe
        B_frame_start = framenumbers[framenumbers.B_frame >= n].index.min()
        continuing = []
        for i, f in enumerate(piv_files[B_frame_start:end_file]):
            start_time = time.time()
            # read frame numbers from PIV output
            fn, A_img_no, B_img_no = PIV_framenumbers(f)

            # read PIV data, ignoring extra columns if they exist
            data = pd.read_csv(f, skiprows=3, usecols=[0, 1, 2, 3],
                               names=['x', 'y', 'u', 'v'])

            # fill nan with zeros, do not use if using fill_array option
            if replace_nan:
                data.loc[np.isnan(data.u)].u = 0.
                data.loc[np.isnan(data.v)].v = 0.

            # calcuate 2d mapts of displacement values
            U = gridize(data.x, data.y, data.u)
            V = gridize(data.x, data.y, data.v)

            # fill array nans with nearest valid neighbors
            if fill_array:
                U = fill(U)
                V = fill(V)

            # calculate shear strain
            dUdY = np.gradient(U.T)[0]
            dVdX = np.gradient(V.T)[1]
            ep_xy = 0.5 * (dUdY + dVdX)
            data['exy'] = degridize(ep_xy)[:, 2]

            # calcuate vorticity (Meynart, Fung, Eringen, Timoshekno, etc...)
            VO = dVdX - dUdY
            data['vort'] = degridize(VO)[:, 2]

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
                                 names=['x', 'y', 'temp_c'])

            # place temps in data by merging to align with existing values
            data = data.merge(temp_e, left_on=['x', 'y'], right_on=['x', 'y'],
                              how='left')
            # this is not needed if already masked in PIVlab
            if mask_piv:
                xy = np.array([data.x, data.y]).T
                # read surface
                surface = pd.read_hdf(
                    surface_hdf5, 'wedgetop_%05.0f' % B_img_no)
                surface.x = surface.x-xmin
                surface.y = surface.y-(images.frame_shape[0] - ymax)

                # return boolean index of points beneath surface
                subsurface_piv = find_below(surface, xy)

                # remove background velocities
                data = data.loc[subsurface_piv]
            # background displacement coordinate values for kd-tree
            xy_target = np.vstack((data.x, data.y)).T

            # INITIALIZE particle displacement arrays
            if i == 0:
                # initial particle location grid for kd-tree
                xy_orig = parts[['x', 'y']].values

                # perform kd-tree, only using nearest PIV grid point
                dist, ind = do_kdtree(xy_target, xy_orig, 1)

                # find associated displacements, strains, and vorticity
                #   within radius
                u, v, exy, vort, e_temp = np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig))
                close_ind = []
                for k in range(len(xy_orig)):
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts) >= 1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig)):
                    if close_ind[k] != None:  # ignore empty
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
                first = pd.DataFrame(np.vstack((first_frame, first_pnums,
                                                nxs, nys, nexy, nvort, ne_temp, nxs_dfm)).T,
                                     columns=['frame', 'particle', 'x', 'y',
                                              'exy', 'vort', 'e_temp',
                                              'x_df'])

            # CALCULATE incremental displacement arrays for each particle
            else:
                # updated particle location grid for kd-tree
                xy_orig_update = np.array([nxs, nys]).T

                # updated kd-tree for new particle locations, using one point
                dist, ind = do_kdtree(xy_target, xy_orig_update, 1)

                # calcuate displacements, strains, and vorticity within radius
                u, v, exy, vort, e_temp = np.zeros(len(xy_orig_update)),\
                    np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig)),\
                    np.zeros(len(xy_orig))
                close_ind = []
                for k in range(len(xy_orig_update)):
                    # determine location of nearest grid points
                    close_pts = ind[k][dist[k] < radius]
                    if len(close_pts) >= 1:
                        close_ind.append(int(close_pts))
                    else:
                        close_ind.append(None)
                for k in range(len(xy_orig_update)):
                    if close_ind[k] != None:  # ignore empty
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
                continuing_df = pd.DataFrame(np.vstack((continuing_frame,
                                                        continuing_pnums, nxs, nys, nexy, nvort,
                                                        ne_temp, nxs_dfm)).T,
                                             columns=['frame', 'particle', 'x', 'y',
                                                      'exy', 'vort', 'e_temp', 'x_df'])
                continuing.append(continuing_df)
            if verbose:
                frame_elapsed = time.time()-start_time
                total_time += frame_elapsed
                num_displaced = len([i for i in close_ind if i is not None])
                print('Calcuated displacements for %g ' % (num_displaced) +
                      'particles for File %05.0f, ' % (fn) +
                      'starting at Frame %05i, ' % (n) +
                      'Frames %05.0f and %05.0f ' % (A_img_no, B_img_no) +
                      'in %06.3f seconds, ' % (frame_elapsed) +
                      '%010.3f total seconds elapsed.' % (total_time))

        # compile data (may be large, >20GB)
        # keep NaNs to filter later, if needed
        df.append(pd.concat((first, pd.concat(continuing))))
        # count of particles as they are iteratively added
        rolling_particle_nos += len(parts.particle.unique())
    return pd.concat(df).reset_index(drop=True)
