from __future__ import division
import os
from os.path import join
import sys
import pandas as pd
import numpy as np
import cv2


# EYE DATA/TASK PARAMETERS
sampleHz = 60;					# sample rate of the eye-tracking data
image_dur = 5000;				# duration (ms) of each image presentation
screenSize_mm = (340, 270)  	# screen size (mm)
screenSize_px = (1280, 1024) 	# screen size (px)
px2mm = screenSize_mm[0]/screenSize_px[0]	# calculate pixel to mm scaling factor


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### --------- Filtering Classes --------------------------
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FixationFilter_IDT():
	"""
	Dispersion Threshold Filter (based on Salvucci et al. (2000)) for definging
	fixations in eye-tracking data
	"""
	def __init__(self):
		# Dispersion parameters
		self.minDuration = 100  	# minimum fixation duration (ms)
		self.maxDispersion = 1.5    # maximum dispersion (degrees of vis angle)
		
		global px2mm				# grab the pixel to mm conversion factor
		self.px2mm = px2mm

		self.fixCounter = 1 # initialize fixation counter for labeling
		
	def applyFilter(self, df):
		"""
		Apply the filter to the input dataframe, representing a trial's eye-tracking data. 
		Return the dataframe with fixation labels column included
		"""
		n_tmpts = df.shape[0]

		# initialize fixNumber array (this will store corresponding fix number for each datapt)
		fixNumber = np.zeros(shape=(n_tmpts))

		start = 0
		end = 0
		while end+1 < n_tmpts:
			
			# skip invalid timepts
			if (df.validity.iloc[start] == 0) and (df.validity.iloc[end] == 0):

				# timestamp at start of window
				startTS = df.timeStamp.iloc[start]

				# build window to cover the min duration threshold
				while end+1 < n_tmpts and ((df.timeStamp.iloc[end] - startTS) < self.minDuration) and (df.validity.iloc[end] == 0):
					end += 1

				winDuration = df.timeStamp.iloc[end] - df.timeStamp.iloc[start]
				if winDuration >= self.minDuration:
					
					# calculate dispersion over current window
					disp = self.dispersion(df, start, end)

					# if less than threshold
					if (disp < self.maxDispersion) and (end+1 < n_tmpts):
						
						# increase window til it passes threshold
						while (self.dispersion(df, start, end) < self.maxDispersion) and (end+1 < n_tmpts) and df.validity.iloc[end+1] == 0:
							end += 1
						
						fixDuration = df.timeStamp.iloc[end] - startTS

						# mark all of these points as fixations, assign appropriate number
						fixNumber[start:end] = self.fixCounter
						self.fixCounter += 1

						# reset values
						end += 2   # skip one to insert saccade
						start = end
					else:
						start += 1
				else:
					start+=1
					end+=1
			else:
				start += 1
				end += 1
		
		# add the fixation number label to original array
		fixNumber[fixNumber == 0] = np.nan  # set fixNumber=0 to nan
		df['fixNumber'] = fixNumber  
		return df
	
	def dispersion(self, df, winStart, winEnd):
		"""
		Calculate the X,Y dispersion over the given window
		Return dispersion expressed in visual angle
		"""
		# get X,Y values for this window
		xVals = df['gaze-X'].iloc[winStart:winEnd+1]
		yVals = df['gaze-Y'].iloc[winStart:winEnd+1]
		disp = (xVals.max()-xVals.min() + yVals.max()-yVals.min())
		
		# convert from px to mm
		disp = disp * self.px2mm

		# convert from mm to visual angle
		dist = np.mean(df['eyeDistance'].iloc[winStart:winEnd+1])   # mean eye-distance over window (mm)
		disp = np.rad2deg(2 * np.arctan((disp/2)/dist))
		
		return disp




### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### --------- Filtering Functions ------------------------
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def filterRaw(trial_df):
	"""
	input: dataframe of raw trial eye-tracking data

	Filtering steps: 
		- remove invalid timepts, fixation cross timepoints
		- interpolate over missing values
		- translate gaze pts to be relative to image

	return: filtered dataframe for trial
	"""

	# grab the X,Y values for fixation cross
	fixCrossX = int(trial_df['fix-X'].iloc[0])
	fixCrossY = int(trial_df['fix-Y'].iloc[0])

	# isolate the timepoints during the "image" portion of each trial
	trial_df = trial_df[trial_df.stimType == 'image']
	
	# translate gaze coordinates to be relative to the stimulus (instead of screen)
	stimLocation = [trial_df['rect-X1'].iloc[0], trial_df['rect-Y1'].iloc[1]] # [x1,y1] of stimulus on screen
	trial_df.loc[:,'gaze-X'] = trial_df.loc[:,'gaze-X'] - stimLocation[0]
	trial_df.loc[:,'gaze-Y'] = trial_df.loc[:,'gaze-Y'] - stimLocation[1]

	# do the same for the fixation cross location col
	trial_df.loc[:,'fix-X'] = fixCrossX - stimLocation[0]
	trial_df.loc[:,'fix-Y'] = fixCrossY - stimLocation[1]

	# set invalid gaze pts to None
	trial_df.loc[trial_df.validity > 0, ['gaze-X','gaze-Y','eyeDistance']] = np.nan

	# maximum window size (in ms) to interpolate over
	max_interp_window = 50 # conservative estimate of blink length  

	# convert interp_window from ms to # of samples
	interp_limit = np.floor(max_interp_window/(1/sampleHz*1000))

	# fill in missing values within window limits
	trial_df.fillna(method='ffill', limit=interp_limit, inplace=True)

	# update the valid datapts column to reflect new values
	trial_df.loc[~np.isnan(trial_df['gaze-X']), 'validity'] = 0

	return trial_df


def defineFixations(trial_df):
	"""
	input: dataframe of (ideally filtered) trial eye-tracking data

	output: input dataframe with added column indicating whether each datapt belongs
	to a "fixation" or not (if so, the value in this column reflects the ordinal value
	of that particular fixation)
	"""

	# create instance of filter object
	fixFilter = FixationFilter_IDT()

	# pass dataframe through the filter
	trial_df = fixFilter.applyFilter(trial_df)

	return trial_df


def summarizeFixations(trial_df):
	"""
	input: dataframe of trial eye-tracking data with column indicating fixation

	output: dataframe with each row reprsenting a single fixation during that trial,
	and columns representing the different dependent variables calculated on each fixation
	"""
	
	# grab list of unique fixation numbers in this trial
	fixLabels = trial_df.fixNumber.unique()

	# loop through each unique label
	for i,fixNum in enumerate(fixLabels[~np.isnan(fixLabels)]):
		
		# pull out the rows for this fixation
		fix_df = trial_df.loc[trial_df.fixNumber == fixNum, :]
		
		# calculate fixation duration
		fixDuration = fix_df.timeStamp.iloc[-1] - fix_df.timeStamp.iloc[0]
		
		# calculate centroid location for fixation
		fixPosX, fixPosY = np.ceil(np.mean(fix_df.loc[:, 'gaze-X':'gaze-Y']))
		
		# calculate the mean eye-distance during this fixation (needed to convert subsequent values from px to vis angle)
		fixEyeDist = np.mean(fix_df.eyeDistance)
			
		# calculate distance from previous fixation
		if i == 0:
			# if its the first fixation, use the fixation cross location
			prevFix_X = fix_df['fix-X'].iloc[0]
			prevFix_Y = fix_df['fix-Y'].iloc[0]
		distX = abs(fixPosX-prevFix_X)
		distY = abs(fixPosY-prevFix_Y)
		distPrev = np.sqrt(distX**2 + distY**2)   # do some hypoteneuse calculating
		distPrev = distPrev * px2mm              # convert distance to mm
		distPrev = np.rad2deg(2 * np.arctan((distPrev/2)/fixEyeDist))  # convert to visual angle
		
		# calculate distance from center of image
		distX = abs(fixPosX-(imW/2))
		distY = abs(fixPosY-(imH/2))
		distCenter = np.sqrt(distX**2 + distY**2)
		distCenter = distCenter * px2mm
		distCenter = np.rad2deg(2 * np.arctan((distCenter/2)/fixEyeDist))
		
		# calculate direction of movement from previous fixation
		dx = fixPosX - prevFix_X
		dy = fixPosY - prevFix_Y
		radsFromPrev = np.arctan2(-dy, dx)   # invert dy to account for screen coordinates going from top to bottom
		radsFromPrev %= 2*np.pi              # convert to radians relative to the +x axis
		dirFromPrev = np.rad2deg(radsFromPrev) # express direction degrees
		
		# update the previous fixation values
		prevFix_X = fixPosX
		prevFix_Y = fixPosY
		
		# figure out the appropriate AOI label (if any)
		fixCoords = (fixPosX, fixPosY)
		AOI_label = trialAOIs.isAOI(fixCoords)
		
		# figure out which quadrant of the image the fixation falls in
		if fixPosX <= imW/2:
			horizHalf = 'left'
		else:
			horizHalf = 'right'
		if fixPosY <= imH/2:
			vertHalf = 'top'
		else:
			vertHalf = 'bot'
			
		# write output to new dataframe
		fixSummary = pd.DataFrame({'fixNumber': fixNum,
								 'duration':fixDuration,
								 'fixPosX': fixPosX,
								 'fixPosY': fixPosY,
								 'crossX': fix_df['fix-X'].iloc[0],
								 'crossY': fix_df['fix-Y'].iloc[0],
								 'distFromPrev': distPrev,
								 'dirFromPrev':dirFromPrev,
								 'distFromCenter':distCenter,
								 'AOI': AOI_label,
								 'vertHemi': vertHalf,
								 'horizHemi': horizHalf}, index=[0])
		if i == 0:
			allFix_df = fixSummary
		else:
			allFix_df = pd.concat([allFix_df, fixSummary], ignore_index=True)

	# ADD BASIC TRIAL INFO (NUMBER, IMAGE NAME, ETC)

	return allFix_df












