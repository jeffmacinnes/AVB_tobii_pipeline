import sys
import os
import argparse
import filtering_utils as filt
from os.path import join
import pandas as pd
import numpy as np


def processData(inputDataPath):
	"""
	Given the path to a subject's raw data file (e.g. AVB_XXX_eyeData.txt), run the 
	data through the various stages of cleanup: filtering the data,  defining fixations, summarizing fixations, summarizing trial
	"""

	### Define paths
	fileParts = inputDataPath.split('/')
	SUBJ = fileParts[-2]
	OUTPUT_dir = ('/').join(fileParts[:-1])

	EXP_dir = ('/').join(fileParts[:-3])
	STIM_dir = join(EXP_dir, 'Stimuli')

	# Read raw data as dataframe
	raw_df = pd.read_table(inputDataPath)

	# loop through all unique trials
	for i, trial in enumerate(np.unique(raw_df['trialNum'])):

		# grab trial data
		trial_df = raw_df[(raw_df.trialNum == trial)]

		#####################################
		### Step 1: Filter Raw Data
		#####################################
		trial_filtered = filt.filterRaw(trial_df)

		
		#####################################
		### Step 2: Define Fixations on filtered dataframe
		#####################################
		trial_filtered = filt.defineFixations(trial_filtered)


		#####################################
		### Step 3: Summarize Each Fixation
		#####################################
		
		# check if there's an AOI for this trial
		stimName = trial_filtered.imageName.iloc[0]
		print stimName
		AOI_name = stimName.split('/')[1][:-4] + '_AOIs.png'
		AOI_path = join(STIM_dir, 'AOIs/finished_AOIs', AOI_name)
		if not os.path.exists(AOI_path):
			AOI_path = None

		# summarize fixations (label AOIs, if applicable)
		trial_fixations= filt.summarizeFixations(trial_filtered, AOI_path)


		#####################################
		### Step 4: Summarize Trial Overall
		#####################################
		trial_summary = filt.summarizeTrial(trial_fixations)


		#####################################
		### Store this trial's output in master dataframes
		#####################################
		if i == 0:
			allTrials_filtered = trial_filtered.copy()
			allTrials_fixations = trial_fixations.copy()
			allTrials_summaries = pd.DataFrame(trial_summary).T        # convert trial_summary from series to dataframe
		else:
			allTrials_filtered = pd.concat([allTrials_filtered, trial_filtered], ignore_index=True)
			allTrials_fixations = pd.concat([allTrials_fixations, trial_fixations], ignore_index=True)
			allTrials_summaries = pd.concat([allTrials_summaries, pd.DataFrame(trial_summary).T], ignore_index=True)


	### Write all output to the disk for this subject
	# filtered data
	allTrials_filtered.to_csv(join(OUTPUT_dir, (SUBJ + '_filtered.tsv')), 
								sep='\t',
								header=True,
								index=False,
								float_format='%.3f')
	
	# fixation data
	fixColOrder = ['trialNum', 'imageName', 'crossX', 'crossY',
					'fixNumber', 'duration','fixPosX', 'fixPosY',
					'dirFromPrev', 'distFromPrev', 'distFromCenter',
					'AOI', 'horizHemi', 'vertHemi']
	allTrials_fixations[fixColOrder].to_csv(join(OUTPUT_dir, (SUBJ + '_fixations.tsv')), 
								sep='\t',
								header=True,
								index=False,
								float_format='%.2f')

	# summary data
	summaryColOrder = ['trialNum', 'imageName', 'leftEye', 'rightEye', 'eyesCombined',
						'nose', 'mouth', 'none', 'nonFixation',
						'left', 'right', 'top', 'bot']
	allTrials_summaries[summaryColOrder].to_csv(join(OUTPUT_dir, (SUBJ + '_trialSummary.tsv')), 
								sep='\t',
								header=True,
								index=False,
								float_format='%.2f')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('raw_data', help="path to raw data text file for this subject")
	args = parser.parse_args()

	processData(args.raw_data)












