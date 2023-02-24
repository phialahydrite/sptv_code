import time
import pandas as pd
import numpy as np
from scipy.ndimage import filters
from util import *


def particle_displacer(piv_files,
                       particles,
                       framenumbers,
                       end_file,
                       radius=24,
                       verbose=False,
                       replace_nan=False,
                       fill_array=False):
    '''                       
    Using Matlab PIvab output with x,y,u,v data [files], displace synthetic 
        markers [particles], while simutaneously filtering PIV data that 
        falls outside of the model wedge, using calcuated surfaces 
        [surface_hdf5]
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

            # flip PIVlab format upside-down and shift on x-axis
            _, _, bound = roi_geom(piv_files[0])
            roixmin = bound.xmin.values
            roiymax = bound.ymax.values
            data.v = -data.v
            data.y = roiymax-data.y
            data.x = data.x - roixmin
            # data.y = data.y.max()-data.y
            # data.x = data.x - x_min

            # background displacement coordinate values for kd-tree
            xy_target = np.vstack((data.x, data.y)).T

            # INITIALIZE particle displacement arrays
            if i == 0:
                # initial particle location grid for kd-tree
                xy_orig = parts[['x', 'y']].values

                # perform kd-tree, only using nearest PIV grid point
                dist, ind = do_kdtree(xy_target, xy_orig, 1)

                # find associated displacements within radius
                u, v = np.zeros(len(xy_orig)),np.zeros(len(xy_orig))
                    
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
                nxs = parts.x.values + u
                nys = parts.y.values + v

                # compile initial dataframe
                first_pnums = np.arange(len(parts.particle.unique())) + \
                    rolling_particle_nos
                first_frame = np.ones(len(first_pnums))*B_img_no
                first = pd.DataFrame(np.vstack((first_frame, first_pnums,
                                                nxs, nys)).T,
                                     columns=['frame', 'particle', 'x', 'y'])

            # CALCULATE incremental displacement arrays for each particle
            else:
                # updated particle location grid for kd-tree
                xy_orig_update = np.array([nxs, nys]).T

                # updated kd-tree for new particle locations, using one point
                dist, ind = do_kdtree(xy_target, xy_orig_update, 1)

                # calcuate displacements, strains, and vorticity within radius
                u, v = np.zeros(len(xy_orig_update)),\
                    np.zeros(len(xy_orig_update))
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
                nxs = nxs + u
                nys = nys + v

                # compile further displaced particle and frame information
                continuing_pnums = np.arange(len(parts.particle.unique())) + \
                    rolling_particle_nos
                continuing_frame = np.ones(len(continuing_pnums))*B_img_no
                continuing_df = pd.DataFrame(np.vstack((continuing_frame,
                                                        continuing_pnums, nxs, nys)).T,
                                             columns=['frame', 'particle', 'x', 'y'])
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
