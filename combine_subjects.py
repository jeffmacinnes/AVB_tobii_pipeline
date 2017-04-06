import os
import sys
import numpy as np 
import pandas as pd 
from os.path import join

"""
Combine the fixations and trialSummaries output files across subjects
"""
# set up input and output paths
pathRoot = '../../Results'
outputDir = join(pathRoot, 'groupResults')
if not os.path.isdir(outputDir):
	os.makedirs(outputDir)

subjNums = np.arange(55,80)
				
fileTypes = {'fixations': '_fixations.tsv', 
			'summaries':'_trialSummary.tsv',
			'filtered': '_filtered.tsv', }

for i,subj in enumerate(subjNums):
	subjName = 'AVB_' + str(subj).zfill(4)
	print subjName

	# loop through the two different within-subject file types
	for thisFileType in fileTypes.keys():

		subj_file_path = join(pathRoot, subjName, (subjName + fileTypes[thisFileType]))

		# check if file exists
		if os.path.exists(subj_file_path):
			# read in dataframe
			df = pd.read_table(subj_file_path)

			# insert subject number into dataframe
			df.insert(0, 'subj', subjName)

			# assign the appropriate output df
			if i == 0:
				if thisFileType == 'fixations':
					allSubj_fixations = df.copy()
				elif thisFileType == 'summaries':
					allSubj_summaries = df.copy()
				elif thisFileType == 'filtered':
					allSubj_filtered = df.copy()
			else:
				if thisFileType == 'fixations':
					allSubj_fixations = pd.concat([allSubj_fixations, df], ignore_index=True)
				elif thisFileType == 'summaries':
					allSubj_summaries = pd.concat([allSubj_summaries, df], ignore_index=True)
				elif thisFileType == 'filtered':
					allSubj_filtered = pd.concat([allSubj_filtered, df], ignore_index=True)

		else:
			print 'No ' + (subjName + fileTypes[thisFileType]) + ' file...'


# Write all outputs
allSubj_fixations.to_csv(join(outputDir, 'group_fixations.tsv'),
					sep='\t',
					header=True,
					index=False)

allSubj_summaries.to_csv(join(outputDir, 'group_summaries.tsv'),
					sep='\t',
					header=True,
					index=False)

allSubj_filtered.to_csv(join(outputDir, 'group_filtered.tsv'),
					sep='\t',
					header=True,
					index=False)



