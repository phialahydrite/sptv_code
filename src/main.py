#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:37:33 2022

@author: Phiala Thouvenin
"""
import glob
import pims
import pandas as pd
import numpy as np
from calc import *
from util import *


'''
Generate particles for one experiment.
Define basic parameters and generate synthetic particle grid
'''
# pjt, high friction, 15deg prebuilt taper, 9ero,glassbeads 1cm
x_min, y_max = 140, 1220
x_max, y_min = x_min+4400, 400
# need to redefine as '../data/pjt_highfric...' if providing sample data
prefix = 'pjt_highfric_15deg_1cmglassbeads_9ero'
images = pims.ImageSequence(f'{prefix}/*.jpg')
h5name = f'{prefix}/pjt_highfric_15deg_1cmglassbeads_9ero_trajs_size11.h5'
surfs = f'{prefix}/pjt_highfric_15deg_1cmglassbeads_9ero_surfnocv.h5'
xmin, xmax, ymin, ymax = 140, x_min+4400, 400, 1220
pixelmm_scale = 3.05
scale = 61.
pfiles = glob.glob('pjt_highfric_15deg_1cmglassbeads_9ero/piv_nomask/*.txt')
pfiles.sort()
fileinfo = [np.array(PIV_framenumbers(f)) for f in pfiles]
fileinfo = pd.DataFrame(fileinfo, columns=['filenumber', 'A_frame', 'B_frame'])
# ero_events = pd.read_excel('pjt_highfric_15deg_1cmglassbeads_9ero/pjt_highfric_15deg_1cmglassbeads_9ero_erosional_events.xsx')
im_w = images.frame_shape[1]
im_h = images.frame_shape[0]
end_cutoff = len(images)
verge = 'single'
# start and skip, 2 anf 4 or most models
frame_start = 2
frame_skip = 4
# erosional temperatures as calculated using erosional_thickness_and_time
efiles = glob.glob(
    'pjt_highfric_15deg_1cmglassbeads_9ero_118kappa_0csurf_temps_cumsum_wh/ero_temps_*.txt')
efiles.sort()
chan_plot = False
exh = True
print('Processing %s' % prefix)

'''
PARTICLE PIV DISPLACEMENT AND GRID GENERATION
'''
# get reigon-of-interest information
roiwidth, roiheight, _ = roi_geom(pfiles[0])

# spacing of particles and frames before adding new particles
part_spacing = 4
frame_spacing = scale
extra = 4
step = 2  # PIV analysis step
begin_file = 0
end_file = int(end_cutoff / step)
begin_frame = begin_file * step
end_frame = end_file * step

# read surface
surface = pd.read_hdf(surfs, 'wedgetop_%05.0f' % begin_frame)
surface.x = surface.x - xmin
surface.y = surface.y - (im_h-ymax)

# initial grid covering entire area within region of interest
Xs = np.linspace(0, roiwidth, round(roiwidth/part_spacing))
Ys = np.linspace(0, roiheight, round(roiheight/part_spacing))
grid = [(x, y) for x in Xs for y in Ys]
grid.sort()
grid = pd.DataFrame(np.array(grid), columns=['x', 'y'])

# return boolean index of points beneath surface
subsurface = find_below(surface, grid.values)
grid = grid[subsurface]
grid['particle'] = list(range(len(grid)))
grid['frame'] = 0

# grid that will be repetitively added each [frame_spacing*extra] frames
# this is at the far end of the model where a gap opens with lateral
# motion of the model
box_w = roiwidth-(pixelmm_scale*frame_spacing*extra)
box_h = roiheight
Xs = np.linspace(box_w, roiwidth, round(
    (pixelmm_scale*frame_spacing)/part_spacing))
Ys = np.linspace(0, box_h, round(box_h/part_spacing))
grid_sel = [(x, y) for x in Xs for y in Ys]
grid_sel.sort()
grid_sel = pd.DataFrame(np.array(grid_sel), columns=['x', 'y'])
grid_sel = grid_sel[grid_sel.y <= 2*scale]
grid_sel['particle'] = np.array(list(range(len(grid_sel)))) + len(grid)
grid_sel['frame'] = 0

# grid on bottom of box that is repetitively added so to not leave gaps
# will be added each [frame_spacing] frames
box_w = 0  # all the way to other end
box_h = 2*scale  # in centimeters
Xs = np.linspace(box_w, roiwidth, round(
    (pixelmm_scale*frame_spacing)/part_spacing))
Ys = np.linspace(0, box_h, round(box_h/part_spacing))
grid_sel_bottom = [(x, y) for x in Xs for y in Ys]
grid_sel_bottom.sort()
grid_sel_bottom = pd.DataFrame(np.array(grid_sel_bottom), columns=['x', 'y'])
grid_sel_bottom['particle'] = np.array(list(range(len(grid_sel_bottom)))) \
    + len(grid) + len(grid_sel)
grid_sel_bottom['frame'] = 0

# add this bottom layer of particles if desired
# later_grids = pd.concat((grid_sel,grid_sel_bottom))
later_grids = grid_sel


# compilation of original grids and these repetitively added grids
added_grids = []
for i in list(range(int(end_frame/frame_spacing)))[1:-1]:
    # further grids
    grid_d = later_grids.copy()
    grid_d.particle = np.array(list(range(len(grid_d)))) + i*len(grid_sel) \
        + i*len(grid_sel_bottom)
    grid_d.frame = i * frame_spacing
    added_grids.append(grid_d)
all_grids = pd.concat((grid, pd.concat(added_grids))).reset_index()
# all_grids = pd.concat(added_grids)

# calcuate particle displacements
# pts_fname = '%s_artificial_pts_addbottomline_temp.csv'%prefix
pts_fname = '%s_artificial_pts_temp.csv' % prefix
# x-values shifted to location of deformation front
if verge == 'single':
    fronts = deformation_front_mode(surfs, end_frame, from_larger=True,
                                    vergence='s')
else:
    fronts = deformation_front_mode(surfs, end_frame, from_larger=True,
                                    vergence='d')
pts = particle_displacer(pfiles, efiles, fronts,
                         all_grids, fileinfo, surfs, 0,
                         end_file, verbose=False, fill_array=True,
                         radius=24)
# cumulative values
pts['cumexy'] = pts.groupby('particle')['exy'].cumsum()
pts['cumvort'] = pts.groupby('particle')['vort'].cumsum()
pts['cum_conv'] = pts.frame * pixelmm_scale
pts.to_csv(pts_fname, chunksize=1000000, float_format='%.8f', index=False)
del pts
