import os
import sys
from os.path import join
import numpy as np

pathRoot = '../../Results'

subjNums = np.arange(55,80)
for subj in subjNums:
	subjName = 'AVB_' + str(subj).zfill(4)
	print subjName

	subj_rawData_path = join(pathRoot, subjName, (subjName + '_eyeData.txt'))

	if os.path.exists(subj_rawData_path):
		try:
			cmd_str = 'python filter_raw_data.py ' + subj_rawData_path
			os.system(cmd_str)
		except:
			print 'Error on subject: ' + subjName
	else:
		print 'No valid raw data file for subject: ' + subjName
