from __future__ import division
import os
from os.path import join
import sys
import pandas as pd
import numpy as np
import cv2

pd.options.mode.chained_assignment = None  # default='warn'


# EYE DATA/TASK PARAMETERS
sampleHz = 60;					# sample rate of the eye-tracking data
stimDuration = 5000;			# duration (ms) of each stim presentation
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
		df.loc[:,'fixNumber'] = fixNumber  
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


class AOIs:
	"""
	Class to work with specified AOI image file, store each AOIs coordinates
	Input:  -2D ndarray containing unique AOI labels by number
			-scaleFactor: scale factor between AOI image and displayed stim size during the task
	"""
	def __init__(self, AOIimage, scaleFactor):

		# calculate scale factor between AOI image and displayed image
		self.scaleFactor = scaleFactor
		self.AOI = AOIimage
		self.isInverted = False

		# flip image if necessary
		if self.isInverted:
			self.AOI = np.flipud(self.AOI) 		# flip the image upside down

		# extract the unique, nonzero, values in this AOI
		self.AOI_codes = np.unique(self.AOI[self.AOI > 0])

		#### CREATE DICT TO STORE EACH AOIS COORDS
		self.AOIs = {}				# dicitionary to store all of the AOIs and coordinates
		for val in self.AOI_codes:
			self.this_AOI = self.AOI == val 					# make a unique image for this value only
			self.Xcoords = np.where(self.this_AOI==True)[1]		# pull out x-coordinates for this AOI (NOTE: remember, (row,col) convention means (y,x))
			self.Ycoords = np.where(self.this_AOI==True)[0]		# pull out y-coordinates for this AOI
			self.coords = [(self.Xcoords[x], self.Ycoords[x]) for x in range(len(self.Xcoords))]	# convert to list of tuples

			# map values to names
			print(val)
			if val == 64:
				self.AOI_name = 'rightEye'
			elif val == 128:
				self.AOI_name = 'leftEye'
			elif val == 191:
				self.AOI_name = 'nose'
			elif val == 255:
				self.AOI_name = 'mouth'
			else:
				print('AOI image has value of: ' + str(val) + '. Not found in key.')

			# store names and coordinates in dictionary (NOTE: coordinates are stored as (x,y) pairs...so: (column, row))
			self.AOIs[self.AOI_name] = self.coords


	def isAOI(self, coordinates):
		"check if the specified coordinates fall into one of the AOIs"

		# scale coordinates
		self.x = np.round(coordinates[0] * self.scaleFactor) #.astype('uint8')
		self.y = np.round(coordinates[1] * self.scaleFactor) #.astype('uint8')
		self.this_coord = (self.x, self.y)

		# loop through the AOIs in the dictionary until you find which (if any) it belongs to
		self.found_AOI = False
		for name, coords in self.AOIs.iteritems():
			if self.this_coord in coords:
				self.AOI_label = name
				self.found_AOI = True
				break

		if not self.found_AOI:
			self.AOI_label = 'none'

		return self.AOI_label


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
	max_interp_window = 100 # conservative estimate of blink length  

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


def summarizeFixations(trial_df, AOI_path):
	"""
	input: -dataframe of trial eye-tracking data with column indicating fixation
 		   -path to AOI file for this image

	output: dataframe with each row reprsenting a single fixation during that trial,
	and columns representing the different dependent variables calculated on each fixation
	"""
	# get stim dimensions for this trial's stimulus
	imW = trial_df.loc[trial_df.stimType == 'image','rect-X2'].iloc[0] - trial_df.loc[trial_df.stimType == 'image','rect-X1'].iloc[0]
	imH = trial_df.loc[trial_df.stimType == 'image','rect-Y2'].iloc[0] - trial_df.loc[trial_df.stimType == 'image','rect-Y1'].iloc[0]

	if AOI_path is not None:
		stimAOI = cv2.imread(AOI_path)	 # read AOI file as ndarray
		stimAOI = stimAOI[:,:,0]         # grab first color channel only (other 2 are redundant)
		AOI_scaleFactor = stimAOI.shape[1]/imW   # scale factor for mapping gaze coords to AOI dimensions (note: cv2.imread returns shape as [h,w])
		
		# instantiate AOI object
		trialAOIs = AOIs(stimAOI, AOI_scaleFactor)

	# check if there are any valid fixations during this trial
	if np.nansum(trial_df.fixNumber) > 0:
		
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
			if AOI_path is not None:
				fixCoords = (fixPosX, fixPosY)
				AOI_label = trialAOIs.isAOI(fixCoords)
			else:
				AOI_label = 'none'
			
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
			fixSummary = pd.DataFrame({'trialNum': trial_df.trialNum.iloc[0],
									 'imageName': trial_df.imageName.iloc[0],
									 'fixNumber': fixNum,
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
		
	# if no valid fixations in trial, return dataframe of nans
	else:
		allFix_df = pd.DataFrame({'trialNum': trial_df.trialNum.iloc[0],
								 'imageName': trial_df.imageName.iloc[0],
								 'fixNumber': np.nan,
								 'duration':np.nan,
								 'fixPosX': np.nan,
								 'fixPosY': np.nan,
								 'crossX': trial_df['fix-X'].iloc[0],
								 'crossY': trial_df['fix-Y'].iloc[0],
								 'distFromPrev': np.nan,
								 'dirFromPrev':np.nan,
								 'distFromCenter':np.nan,
								 'AOI': np.nan,
								 'vertHemi': np.nan,
								 'horizHemi': np.nan}, index=[0])

	# return df with all fixations summarized for this trial
	return allFix_df


def summarizeTrial(fix_df):
	"""
	Summarize all fixations in a single trial.
	input: dataframe containing the summarized fixations for this trial

	output:  series representing various fixations summaries within that trial 
	(e.g. proportion of time spent looking at the left eye)
	"""
	totalFixDuration = fix_df.duration.sum()  # total time spent on fixations within the trial

	# sum fixation duration by AOI type
	summedDurationByFix = fix_df.groupby('AOI').duration.sum()

	# calculate a proportion of trial time spent in each AOI
	calcProportion = lambda x: x/stimDuration
	trialSummary = summedDurationByFix.apply(calcProportion)

	# add proportions for fixation time spent in image halves (top vs bot; left vs right)
	for hemi in ['vertHemi', 'horizHemi']:
		summedDurationByHemi = fix_df.groupby(hemi).duration.sum()
		hemiProps = summedDurationByHemi.apply(calcProportion)
		trialSummary = pd.concat([trialSummary, hemiProps])
	
	# make sure all possible AOIs are represented
	for aoi in ['leftEye', 'rightEye', 'nose', 'mouth', 'none', 'left', 'right', 'top', 'bot']:
		if aoi not in trialSummary.index:
			trialSummary.loc[aoi] = 0.0

	# add entry for proportion of time that WAS NOT a fixation
	trialSummary.loc['nonFixation'] = (stimDuration-totalFixDuration)/stimDuration

	# add entry for fixation proportion combined across eyes
	trialSummary.loc['eyesCombined'] = trialSummary.loc['leftEye'] + trialSummary.loc['rightEye']

	# add info about trial number and image name to the output series
	trialSummary.loc['trialNum'] = int(fix_df.trialNum.iloc[0])
	trialSummary.loc['imageName'] = fix_df.imageName.iloc[0]

	# return series with summarized trial info
	return trialSummary













