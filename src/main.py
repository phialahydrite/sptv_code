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
This set of parameters represents one analog model from Thouvenin (2022).
'''
# need to redefine as '../data/pjt_highfric...' if providing sample data
prefix = '../data/pjt_highfric_15deg_glass1cm_071619_crop'
images = pims.ImageSequence(f'{prefix}_*.jpg')
pixelmm_scale = 3.05
scale = 61.
pfiles = glob.glob('../data/github_test_*.txt')
pfiles.sort()
fileinfo = [np.array(PIV_framenumbers(f)) for f in pfiles]
fileinfo = pd.DataFrame(fileinfo, columns=['filenumber', 'A_frame', 'B_frame'])
# ero_events = pd.read_excel('pjt_highfric_15deg_1cmglassbeads_9ero/pjt_highfric_15deg_1cmglassbeads_9ero_erosional_events.xsx')
im_w = images.frame_shape[1]
im_h = images.frame_shape[0]
end_cutoff = len(images)
# start and skip, 2 anf 4 or most models
frame_start = 2
frame_skip = 4

'''
PARTICLE PIV DISPLACEMENT AND GRID GENERATION
'''
# get reigon-of-interest information
roiwidth, roiheight, _ = roi_geom(pfiles[0])

# spacing of particles and frames before adding new particles
part_spacing = 4
frame_spacing = scale
extra = 4
step = 1  # PIV analysis step
begin_file = 0
end_file = int(end_cutoff / step)
begin_frame = begin_file * step
end_frame = end_file * step

# initial grid covering entire area within region of interest
Xs = np.linspace(0, roiwidth, round(roiwidth/part_spacing))
Ys = np.linspace(0, roiheight, round(roiheight/part_spacing))
grid = [(x, y) for x in Xs for y in Ys]
grid.sort()
grid = pd.DataFrame(np.array(grid), columns=['x', 'y'])
grid['particle'] = list(range(len(grid)))
grid['frame'] = 0
# calcuate particle displacements
# pts_fname = '%s_artificial_pts_addbottomline_temp.csv'%prefix
pts_fname = f'{prefix}_artificial_pts_temp.csv'

pts = particle_displacer(pfiles,grid, fileinfo, 0,
                         end_file, verbose=False, fill_array=True,
                         radius=24)

pts.to_csv(pts_fname, chunksize=1000000, float_format='%.8f', index=False)
del pts
