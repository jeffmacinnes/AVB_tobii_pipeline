#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import fnmatch
import shutil
import subprocess

import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import pandas as pd
import seaborn as sns

from skimage import io
from skimage.transform import resize
from os.path import join, split, splitext, isdir

# global vars
inputDir = '../../Results/groupResults'
stimDir = '../../Stimuli'

scale_factor = 1.414		# factor by which the RAW image is scaled before appearing on the screen

#######################---METHODS---####################################
# Graphing helper function
def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


######## SINGLE SUBJECT PLOTS *************************
def mk_fixationPlots(subject_id):
	""" 
	Will make fixation plots for all of the trials found in the specified subject's fixation outputfile
	"""

	# set up paths for this subject
	subj_dir = join(RESULTS_DIR, subject_id)
	subj_plots_dir = join(subj_dir, 'plots')

	# Read in fixation text file (output from filterTools.filterArcim)
	fix_fname = join(subj_dir, (subject_id + '_fixations.txt'))
	fix_df = pd.read_table(fix_fname)

	# figure out how many trials 
	n_trials = np.max(fix_df.trialNumber)

	# loop through all trials
	for trial in np.arange(1, n_trials+1):

		# extract the fixation data for this trial
		this_trial = fix_df[fix_df.trialNumber == trial]

		# if there's fixation data for this trial, submit to plotFixations
		if not this_trial.empty:
			this_fig = plotFixations(this_trial)

			# save this figure
			figName = ('fixations_trial_' + str(trial) + '.png')
			plt.savefig(join(subj_plots_dir, figName))
			plt.close()

			print 'Figure created for ' + subject_id + ' trial ' + str(trial)



def plotFixations(df):
	""" 
	Will create a single plot showing all fixations for the trial specified in df
	"""
	# get the relevant info for this trial from the first row of data
	subjID = df.subjID.iloc[0]
	trialNum = df.trialNumber.iloc[0]
	imageName = df.imageName.iloc[0]
	isInverted = df.isInverted.iloc[0]
	crossX = df.crossX.iloc[0]
	crossY = df.crossY.iloc[0]

	# load in the raw image, and resize to the displayed size 
	bgImage = io.imread(join(stimDir, imageName))
	newHeight = np.floor(bgImage.shape[0]*scale_factor)
	newWidth = np.floor(bgImage.shape[1]*scale_factor)
	bgImage = resize(bgImage, (newHeight, newWidth))

	if isInverted:
		bgImage = np.flipud(bgImage)

	### BEGIN PLOTTING
	if newHeight > newWidth:
		setup_graph(fig_size=(12,15))
	else:
		setup_graph(fig_size=(15,12))
	
	# BG Image
	plt.imshow(bgImage, alpha=0.5)
	plt.axis("off")

	# define colormap
	cmap = plt.get_cmap('Blues', df.shape[0])		# return discrete values from the specified colormap

	# Fixation circles
	plt.scatter(df.fixPosX, df.fixPosY,
				s=df.dwellTime*4,
				c=np.arange(df.shape[0]),
				cmap=cmap,
				alpha=0.7)
	
	# Fixation lines
	plt.plot(df.fixPosX, df.fixPosY, 'k', alpha=0.4, lw=.5)

	# color bar
	plt.colorbar(shrink=.7,
				cmap=cmap,
				orientation='horizontal',
				pad=0.05,
				ticks=(np.linspace(0, df.shape[0]-1, df.shape[0]+1) + .5)
				).ax.set_xticklabels([str(x) for x in np.arange(1, df.shape[0]+1)])

	# fixation location
	plt.axhline(y=crossY, color='orange', ls='--', lw=2)
	plt.axvline(x=crossX, color='orange', ls='--', lw=2)
				
	
	# return a reference to this fig
	return plt.gcf()


######## GROUP LEVEL PLOTS *************************
def mk_heatMaps(datafile):
	"""
	For every trial that exists in the submitted datafile, create a heatmap 
	depicting cumulative gaze points for every subject in the datafile
	"""
	# make output directory if doesn't already exist
	OUTPUT_DIR = join(inputDir, 'heatmaps')
	if not isdir(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	
	# open the combined datafile
	df = pd.read_table(datafile)

	# loop through each trial
	for image in np.unique(df.imageName):

		# get the image name separated from the path and file extension
		imageName = image[:-4].split('/')[-1:]

		# grab just the data for this image
		image_df = df[df.imageName == image]

		# calculate image dimensions
		imageWidth = image_df['rect-X2'].iloc[0]-image_df['rect-X1'].iloc[0]
		imageHeight = image_df['rect-Y2'].iloc[0]-image_df['rect-Y1'].iloc[0]

		# loop through both versions (i.e. inverted and upright) if applicable
		for thisOrientation in np.unique(image_df.isInverted):

			if thisOrientation == 0:
				fig_outputName = (imageName[0] + '.png')
			elif thisOrientation == 1:
				fig_outputName = (imageName[0] + '_inverted.png')

			# isolate the datapoints for this version of the image only
			thisOrientation_df = image_df[image_df.isInverted == thisOrientation]

			# determine number of subjects represented
			n_subjs = len(np.unique(thisOrientation_df.subj))

			# prep thefigure 
			fig = plt.figure()

			title = (fig_outputName[:-4] + '  --  ' + str(n_subjs) + ' subjects')
			fig.suptitle(title, fontsize=20)

			# submit this image df to the plotHeatMap function
			this_fig = plotHeatMap(thisOrientation_df, levels=11, fig=fig, plot_pts=False)

			if imageHeight > imageWidth:
				this_fig.set_size_inches=(12,15)
			else:
				this_fig.set_size_inches=(15,12)

			# save this figure
			plt.savefig(join(OUTPUT_DIR, fig_outputName))
			plt.close()

			print 'Figure created for image ' + fig_outputName
	print 'ALL DONE WITH MK_HEATMAPS.........'

def mk_heatMaps_by_fixNumber(datafile):
	"""
	For every unique image in the submitted datafile, loop through the first
	N fixations (following the first fixation due to fixation cross), and create
	a unique heatmap for each one
	"""
	# Settings
	N = 5			# how many fixations do you want to look at

	# make output directory if doesn't already exist
	OUTPUT_DIR = join(GROUP_DIR, 'heatmapsByFixation')
	if not isdir(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	
	# open the combined datafile
	df = pd.read_table(join(GROUP_DIR, datafile))
	#df = df.drop(['timeStamp', 'stimType', 'fix-X', 'fix-Y', 'validity'], axis=1)

	# loop through each unique image in the combined file
	for image in np.unique(df.imageName):

		# get the image name separated from the path and file extension
		imageName = image[:-4].split('/')[-1:]

		# grab the data for this image only
		image_df = df[df.imageName==image]

		# loop through both versions (i.e. inverted and upright) if applicable
		for thisOrientation in np.unique(image_df.isInverted):

			if thisOrientation == 0:
				fig_outputName = (imageName[0] + '.png')
			elif thisOrientation == 1:
				fig_outputName = (imageName[0] + '_inverted.png')

			# isolate the datapoints for this version of the image only
			thisOrientation_df = image_df[image_df.isInverted == thisOrientation]	

			# determine number of subjects represented
			n_subjs = len(np.unique(thisOrientation_df.subjID))

			# prep the master figure to contain all subplots
			fig = plt.figure(figsize=(30,10))
			title = (fig_outputName[:-4] + '  --  ' + str(n_subjs) + ' subjects')
			fig.suptitle(title, fontsize=20)

			# loop through all desired fixations
			for i in range(N):

				# get the data for this fixation number only
				fix_df = thisOrientation_df[thisOrientation_df.fixNumber == i+2]		# add 1 to skip the first (cross related) fixation

				# submit this fixation df to plotHeatMap function
				index = int('1' + str(N) + str(i+1))
				fig = plotHeatMap(fix_df, levels=7, fig=fig, index=index, plot_pts=True)

			# add title to each axes
			for i,ax in enumerate(fig.get_axes(), start=2):
				ax.set_title(('fixation ' + str(i)))

			# save this figure
			fig.savefig(join(OUTPUT_DIR, fig_outputName))
			plt.close()

			print 'Figure created for image ' + fig_outputName[:-4]
	
	print 'ALL DONE WITH MK_HEATMAPS_BY_FIXNUMBER.........'


def mk_heatMaps_by_time(datafile):
	"""
	For every unique image in the submitted datafile, loop through the first N time bins, 
	and create a unique heatmap for each one
	"""
	# Settings
	binSize = 500 			# size of each bin (in ms)
	delay = 200 			# how long after the trial start to begin grouping bins (this is too avoid the first "fixation cross" fixation)
	nBins = 6
	
	# make output directory if doesn't already exist
	OUTPUT_DIR = join(GROUP_DIR, 'heatmapsByTime')
	if not isdir(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	
	# open the combined datafile
	df = pd.read_table(join(GROUP_DIR, datafile))

	# loop through each unique image in the combined file
	for image in np.unique(df.imageName):
		
		# get the image name separated from the path and file extension
		imageName = image[:-4].split('/')[-1:]

		# grab the data for this image only
		image_df = df[df.imageName==image]

		# loop through both versions (i.e. inverted and upright) if applicable
		for thisOrientation in np.unique(image_df.isInverted):

			if thisOrientation == 0:
				fig_outputName = (imageName[0] + '.png')
			elif thisOrientation == 1:
				fig_outputName = (imageName[0] + '_inverted.png')


			# isolate the datapoints for this version of the image only
			thisOrientation_df = image_df[image_df.isInverted == thisOrientation]

			# reset the time stamp column to be relative to the start of the trial
			ts_diff = np.diff(thisOrientation_df.timeStamp) 		# map out abrupt changes in timestamp values
			for i in range(len(ts_diff)):
				if i == 0:
					startTime = thisOrientation_df.timeStamp.iloc[i]

				if abs(ts_diff[i-1]) > 1000: 			# if the timestamp jumped by more than 1000 ms, consider it a new subject
					startTime = thisOrientation_df.timeStamp.iloc[i]

				# subtract the reference time from each subsequent timepoint for this intance
				thisOrientation_df.iloc[i, thisOrientation_df.columns == 'timeStamp'] = thisOrientation_df.iloc[i, thisOrientation_df.columns == 'timeStamp'] - startTime


			# determine number of subjects represented
			n_subjs = len(np.unique(thisOrientation_df.subjID))

			# prep the master figure to contain all subplots
			fig = plt.figure(figsize=(30,10))
			title = (fig_outputName[:-4] + '  --  ' + str(n_subjs) + ' subjects')
			fig.suptitle(title, fontsize=20)

			# loop through each timeBin
			for b in range(nBins):
				binStart = delay + (b*binSize)
				binEnd = binStart + binSize

				# isolate the timepoints for this bin
				bin_df = thisOrientation_df[(thisOrientation_df.timeStamp > binStart) &
									(thisOrientation_df.timeStamp <= binEnd)]

				# submit this fixation df to plotHeatMap function
				index = int('1' + str(nBins) + str(b+1))
				fig = plotHeatMap(bin_df, levels=7, fig=fig, index=index, plot_pts=True)

			# add title to each axes
			for i,ax in enumerate(fig.get_axes()):
				binStart = delay + (i*binSize)
				binEnd = binStart + binSize
				ax.set_title(('bin ' + str(binStart+1) + ':' + str(binEnd)))

			# save this figure
			fig.savefig(join(OUTPUT_DIR, fig_outputName))
			plt.close()

			print 'Figure created for image ' + fig_outputName[:-4]

	print 'ALL DONE WITH MK_HEATMAPS_BY_TIME.........'


def plotHeatMap(df, levels=10, fig=None, index=111, plot_pts=False):
	"""
	create a single heatmap for the data specified in df

	levels: how many levels in the contour (i.e. heat) map?

	fig: specify a figure instance to write into. If no fig specified, will start one

	index: subplot index (will default to 111 if not otherwise specified)

	plot_pts: option to plot individual gaze pts on top of heatmap (default is False)
	"""
	imageWidth = df['rect-X2'].iloc[0]-df['rect-X1'].iloc[0]
	imageHeight = df['rect-Y2'].iloc[0]-df['rect-Y1'].iloc[0]
	imageName = df.imageName.iloc[0]
	isInverted = df.isInverted.iloc[0]

	# filter out any datapts where the gaze position was NOT on the image
	df = df[(df['gaze-X'] >= 0) &
			(df['gaze-X'] <= imageWidth) &
			(df['gaze-Y'] >= 0) &
			(df['gaze-Y'] <= imageHeight)]	


	# load image
	bgImage = io.imread(join(stimDir, imageName))
	bgImage = resize(bgImage, (imageHeight, imageWidth))

	if isInverted:
		bgImage = np.flipud(bgImage)

	### BEGIN PLOTTING
	if fig is None:
		fig = plt.figure()

	ax = fig.add_subplot(index)
	
	# BG Image
	plt.imshow(bgImage, alpha=0.5)
	plt.axis("off")
	ax.set_xlim([0, imageWidth])
	ax.set_ylim([imageHeight, 0])

	# KERNEL DENSITY PLOT 
	sns.set_style("white")
	sns.despine(trim=True)
	sns.kdeplot(df['gaze-X'],
					df['gaze-Y'],
					shade=True,
					shade_lowest=False,
					cmap='viridis',
					n_levels=levels,
					alpha=0.7)
	
	if plot_pts:
		# plot raw data points
		sns.reset_orig()
		plt.plot(df['gaze-X'], df['gaze-Y'], 'r+')
	
	# return a reference to this fig
	return plt.gcf()


if __name__ == '__main__':
	data_file = join(inputDir, 'group_filtered.tsv')
	mk_heatMaps(data_file)







