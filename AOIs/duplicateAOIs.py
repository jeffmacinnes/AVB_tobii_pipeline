"""
The stimuli that begin with the prefix 'hmnBG' had the AOIs drawn on the unfiltered versions of each stimuli (for easier drawing)

However, subsets of these stimuli were passed through spatial filtering (either high pass or low pass) and included in the stimuli set shown to participants during the eye-tracking study. 

To create AOIs for the filtered versions of those stimuli, we are simply copying the AOIs drawn on the nonfiltered and renaming them accordingly
"""

import os
import sys
import shutil
import glob
from os.path import join
import argparse


def duplicateAOIs(AOIdir):
	# find all of the hmnBG AOIs
	searchPattern = join(AOIdir, 'hmnBG_*_AOIs.png')
	sourceAOIs = glob.glob(searchPattern)

	for thisAOI in sourceAOIs:
		AOIdir, AOIname = thisAOI.split('/')

		for prefix in ['hp_', 'lp_']:
			shutil.copyfile(join(AOIdir, AOIname), join(AOIdir, (prefix + AOIname)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('AOIdir', help="path to dir containing finalized AOIs")

	duplicateAOIs(args.stimList)