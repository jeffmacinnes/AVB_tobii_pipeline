import os
import pandas as pd
import numpy as np
import argparse
from os.path import join

"""
For the specified student, read in their stim list, and create the group
files containing data from those stimuli only. 

There is expected to be in the current working directory a stimlist named
like: <student>Stims.txt containing a single column with stimuli paths

Output files will be written to 'artVisionBrain/Analysis/2016_studentData/<studentName>'
"""
cwd = os.getcwd()

def getData(stimList):

	# Read in student's stim list (in current directory)
	studentStims = np.genfromtxt(stimList, dtype='string')

	# define paths
	fileParts = stimList.split('/')
	studentName = fileParts[-1].split('.')[0][:-5]
	EXP_dir = ('/').join(fileParts[:-4])
	RESULTS_dir = join(EXP_dir, 'Results/groupResults')
	OUTPUT_dir = join(EXP_dir, 'Analysis/2016_studentData', studentName)
	if not os.path.isdir(OUTPUT_dir):
		os.makedirs(OUTPUT_dir)

	# Loop through the relevant group data files
	for groupFile in ['group_fixations.tsv', 'group_summaries.tsv']:

		# load the datafile
		group_df = pd.read_table(join(RESULTS_dir, groupFile))

		# create boolean row index of where matching stims occur
		rowIdx = [x in studentStims for x in group_df.imageName]

		# create new dataframe of matching stims
		student_df = group_df.loc[rowIdx, :]

		# write to disk
		output_fname = join(OUTPUT_dir, (studentName + groupFile[5:]))
		student_df.to_csv(output_fname, sep='\t', index=False)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('student', help="first name (lowercase) of student")
	args = parser.parse_args()

	cwd = os.getcwd()
	stimList = join(cwd, (args.student + 'Stims.txt'))
	if not os.path.exists(stimList):
		print 'No stim list file available for student: ' + student
	else:
		getData(stimList)







