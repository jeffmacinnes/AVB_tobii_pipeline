from __future__ import division

import sys
import os
import fnmatch
import shutil
import subprocess
import cv2

import numpy as np
import pandas as pd
from math import atan2, pi, degrees

from skimage import io
from skimage.transform import resize
from os.path import join, split, splitext, exists


# EYE DATA PARAMETERS
sampleHz = 60;					# sample rate of the eye-tracking data
image_dur = 5;					# duration (in sec) of each image presentation

# FILTER CRITERIA
saccThresh = 2 			# pixels/ms for thresholding saccades
minFixDuration = 100	# minimum duration (in ms) to count as fixation
distThresh = 37			# miniumum distance (in px) between 2 consequtive clusters
timeGapThresh = 200     # max time (in ms) between 2 consequetive clusters in order to be called the same 

minDataPts = np.round(sampleHz * image_dur * .2)	# ensure there are a minimum of 20% of the expected number of datapts


def filterRaw(trial_df):
	"""
	input: dataframe of raw trial eye-tracking data

	Filtering steps: 
		- remove invalid timepts, fixation cross timepoints
		- interpolate over missing values
		- translate gaze pts to be relative to image

	return: filtered dataframe for trial
	"""

	# isolate the timepoints during the "image" portion of each trial
	trial_df = trial_df[triad_df.stimType == 'image']

	### Interpolate over mising values
	trial_df.loc[trial_df.validity > 0, 'gaze-X':'gaze-Y'] = None 		# set invalid gaze pts to None
	max_interp_window = 100												# maximum window size (in ms) to interpolate over
	interp_limit = np.floor(max_interp_window/(1/sampleHz*1000))  		# convert interp_window from ms to # of samples

	for col in ['gaze-X', 'gaze-Y']:
		trial_df[col].interpolate(limit=interp_limit,
									limit_direction='both',
									inplace=True)

	### translate gaze coordinates to be relative to the stimulus (instead of screen)
	stimLocation = [trial_df['rect-X1'].iloc[0], trial_df['rect-X1'].iloc[1]]			# [x1,y1] of stimulus on screen
	trial_df['gaze-X'] = trial_df['gaze-X'] - stimLocation[0]
	trial_df['gaze-Y'] = trial_df['gaze-Y'] - stimLocation[1]

	# do the same for the fixation cross location col
	trial_df['fix-X'] = trial_df['fix-X'] - stimLocation[0]
	trial_df['fix-Y'] = trial_df['fix-Y'] - stimLocation[1]

	return trial_df












