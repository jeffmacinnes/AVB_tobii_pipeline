import numpy as np 
import pandas as pd 
import os
from os.path import join

"""
Combine the SIT scores for all subjects into a single text file
"""

# input and output paths
pathRoot = '../../Results'
outputDir = join(pathRoot, 'groupResults')
if not os.path.isdir(outputDir):
	os.makedirs(outputDir)

subjNums = np.arange(55,80)

# start output file
output_f = open(join(outputDir, 'group_SIT.txt'), 'w')
header = '\t'.join(['Subj', 'Total', 'Male', 'Female'])
output_f.write(header + '\n')

# loop through all subjects
for i,subj in enumerate(subjNums):
	subjName = 'AVB_' + str(subj).zfill(4)
	print subjName

	# path to subj SIT scores
	subj_SIT_path = join(pathRoot, subjName, 'SIT_results.txt')

	if os.path.isfile(subj_SIT_path):
		SIT = np.genfromtxt(subj_SIT_path, skip_header=1, delimiter='\t')

		# format these results into a string
		data_string = '\t'.join([subjName, str(SIT[0]), str(SIT[1]), str(SIT[2])])

		# write to output
		output_f.write(data_string + '\n')


	else:
		print 'No SIT file found for subject ' + subjName

# close file
output_f.close()

