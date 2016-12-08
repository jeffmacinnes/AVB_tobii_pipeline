import sys
import os
import argparse
import filtering_utils as filt
from os.path import join


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

		# Filter the raw data for this trial
		trial_filtered = filt.filterRaw(trial_df)

		# add filtered_trial to master 
		if i == 0:
			allTrials_filtered = trial_filtered.copy()
		else:
			allTrials_filtered = pd.concat([allTrials_filtered, trial_filtered], ignore_index=True)

	

	# write filtered data file to disk
	filtered_fname = join(OUTPUT_dir, (SUBJ + '_filtered.csv'))
	allTrials_filtered.to_csv(filtered_fname, 
								sep='\t',
								header=True,
								index=False,
								float_format='%1.0f')


	#####################################
	### Step 2: Define Fixations
	#####################################
	for i, trial in enumerate(np.unique(filtered_df['trialNum'])):

		# grab trial data
		trial_filtered = filtered_df[(filtered_df.trialNum == trial)]

		# Define the fixations on the filtered trial data
		trial_fixations = ff.defineFixations(trial_filtered)

		# add results to master list of all trials
		if i == 0:
			allTrials_fixations = trial_fixations.copy()
		else:
			allTrials_fixations = pd.concat([allTrials_fixations, trial_fixations], ignore_index=True)

	# Write fixation data file to disk
	fixations_fname = join(OUTPUT_dir, (SUBJ + '_fixations.csv'))
	allTrials_fixations.to_csv(fixations_fname, 
								sep='\t',
								header=True,
								index=False,
								float_format='%1.0f')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('raw_data', help="path to raw data text file for this subject")
	args = parser.parse_args()

	processData(args.raw_data)












