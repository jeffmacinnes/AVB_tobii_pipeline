from __future__ import division
import os
import sys
from os.path import join
import numpy as np
import cv2
import pandas as pd
import glob
import matplotlib.pyplot as plt
import json
import argparse




def processAOIs(stimList):
	"""
	Loop through every stimuli in the stimList. For each, 
	combined the individual instances of each AOI. Save output image of 
	combined AOIs. Create a final AOI based on agreement across drawers, 
	and then mask to ensure a given pixel can belong to one, and only one, 
	AOI. Write final AOI output image as a png with distinct pixel values
	corresponding to the different AOIs. 
	"""
	
	### Define Paths
	fileParts =stimList.split('/')
	AOIs_dir = ('/').join(fileParts[:-2])

	# read in list of mappings between stim number and stim names
	stimNames = pd.read_csv(stimList)

	AOIs = ['leftEye', 'rightEye', 'nose', 'mouth']
	imgs = np.arange(stimNames.shape[0])

	# loop through all images
	for img in imgs:
		print 'image: %s' % img

		# get the stim filename for this image, read in image data
		stim_fname = stimNames.imgName[stimNames.imageIdx==img].values[0]
		thisStim = cv2.imread(join(AOIs_dir, 'AOIstims', stim_fname))
		stimDims = thisStim.shape

		# read in stim
		thisStim = cv2.imread(join(AOIs_dir, 'AOIstims', stim_fname))

		######### Sum all different versions of this AOI
		AOIs_summed = {}
		for AOI in AOIs:
			# get list of all different versions of this AOI
			searchPattern = join(AOIs_dir, 'raw_AOIs/*_img' + str(img) + '_' + AOI + '.png')
			subjAOIs = glob.glob(searchPattern)
			
			# make sure at least one match before continuing
			if len(subjAOIs) > 0:
				n_versions = len(subjAOIs)

				# overlay all unique subj drawn AOIs for this AOI
				for i,subjAOI in enumerate(subjAOIs):
					# read in the specific AOI
					thisIm = cv2.imread(subjAOI)

					# convert to grayscale
					thisIm = cv2.cvtColor(thisIm, cv2.COLOR_BGR2GRAY)

					# mask
					thisIm[thisIm > 0] = 1

					# combine
					if i == 0:
						combinedImg = thisIm.copy()
					else:
						combinedImg = combinedImg + thisIm

				# crop combined AOI img to match stim dimensions
				combinedImg = combinedImg[:stimDims[0], :stimDims[1]]

				# write image showing individual subj AOIs overlaid on image
				AOIs_summed[AOI] = combinedImg

		##########################################################
		###### INTERMEDIATE STEP: SAVE FIGS OF THE SUMMED VERSIONS OF ALL INDIVIDUALLY DRAWN AOIS
		###########################################################
		stim_name = stim_fname.split('.')[0]
		
		# show as image
		thisStim = cv2.cvtColor(thisStim, cv2.COLOR_BGR2RGB)
		fig = plt.figure(figsize=(8, 8))
		plt.imshow(thisStim)

		for AOI in AOIs_summed.keys():
			thisAOI = AOIs_summed[AOI]		
			scaleFactor = 255/n_versions								# rescale the values to range [0,255]
			thisAOI = thisAOI * scaleFactor
			thisAOI[thisAOI == 0] = np.nan								# set 0 values to Nan to all bg image to show through
			plt.imshow(thisAOI, cmap='plasma', alpha=.5)				# add each AOI to the plot

		figName = stim_fname.split('.')[0] + '_summedAOIs.pdf'
		cbar = plt.colorbar(ticks=np.linspace(0,255,5))
		cbar.ax.set_yticklabels([str(x) for x in np.arange(1, n_versions+1)])

		plt.savefig(join(AOIs_dir, 'AOI_figs', 'summed_AOIs', figName))
		plt.close(fig)

		##########################################################
		###### PROCESSING STEP: BUILD THE FINAL VERSION OF THE AOIS
		###########################################################
		### remove AOI pixels where less than 2 subjects agreed
		for AOI in AOIs_summed.keys():
			AOIs_summed[AOI][AOIs_summed[AOI]<2] = 0

		####### Reassign overlapping pixels to AOI with more votes
		AOIs_summed_tmp = {}
		for AOI1 in AOIs_summed.keys():
			overlay = AOIs_summed[AOI1]
			for AOI2 in AOIs_summed.keys():
				if AOI1 != AOI2:
					underlay = AOIs_summed[AOI2]
					underlay[overlay>=underlay] = 0	# tie goes to the overlay
					AOIs_summed_tmp[AOI2] = underlay
		AOIs_summed = AOIs_summed_tmp

		for AOI in AOIs_summed.keys():
			if AOI == 'rightEye':
				AOIs_summed[AOI][AOIs_summed[AOI]>0] = 1
			elif AOI == 'leftEye':
				AOIs_summed[AOI][AOIs_summed[AOI]>0] = 2
			elif AOI == 'nose':
				AOIs_summed[AOI][AOIs_summed[AOI]>0] = 3
			elif AOI == 'mouth':
				AOIs_summed[AOI][AOIs_summed[AOI]>0] = 4 


		##### Write these plots showing the final AOIs overlaid on the background stim
		# prep background image
		thisStim = cv2.imread(join(AOIs_dir, 'AOIstims', stim_fname))
		thisStim = cv2.cvtColor(thisStim, cv2.COLOR_BGR2RGB)
		fig = plt.figure(figsize=(10, 10))
		plt.imshow(thisStim)

		colors = {'leftEye':'seismic', 'rightEye':'summer', 'nose':'spring', 'mouth':'autumn'}
		scaleFactor = 255/4	

		# add each AOI to the plot
		for AOI in AOIs_summed.keys():
			thisAOI = AOIs_summed[AOI]
			thisAOI = thisAOI * scaleFactor
			thisAOI[thisAOI == 0] = np.nan
			
			plt.imshow(thisAOI, cmap=colors[AOI], alpha=.7)

		figName = stim_fname.split('.')[0] + '_processedAOIs.pdf'
		plt.savefig(join(AOIs_dir, 'AOI_figs', 'processed_AOIs', figName))
		plt.close(fig)


		##########################################################
		###### FINAL STEP: WRITE THE FINAL VERSION OF THE AOI AS PNG
		##########################################################
		final_AOI = np.zeros(shape=(stimDims[0], stimDims[1]))
		for AOI in AOIs_summed.keys():
			AOI_val = AOIs_summed[AOI].max()

			# rescale the AOI value to be between [0,255]
			thisAOI = AOIs_summed[AOI] * scaleFactor

			# add it to the composite AOI
			final_AOI = final_AOI + thisAOI
		
		# write the final image to the disk
		AOI_output_name = stim_fname.split('.')[0] + '_AOIs.png'
		cv2.imwrite(join(AOIs_dir, 'finished_AOIs', AOI_output_name), final_AOI) 


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('stimList', help="path to stimuli list containing the mapping between image index in AOIdraw and the path to the stimuli")
	args = parser.parse_args()

	processAOIs(args.stimList)



	
	