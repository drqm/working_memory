proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
scripts_dir = '/projects/' + proj_name + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

import mne
import numpy as np
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats
from mne.datasets import sample
from mne.stats import spatio_temporal_cluster_1samp_test
import os
import pickle
from copy import deepcopy
from sys import argv
from src.group_stats import do_stats
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
# src_sample = mne.read_source_spaces(sample_subjects_dir +
#                                     '/fsaverage/bem/fsaverage-vol-5-src.fif')
src_sample = mne.read_source_spaces(subs_dir +
                                    'fsaverage/bem/fsaverage-vol-5-src.fif')
stats_dir = wdir + 'results/stats/'

# Load and morph a source time course
dfname = data_dir + '0021_LZW_TFR_src2_delta.p'
dfile = open(dfname,'rb')
all_stc = pickle.load(dfile)
dfile.close()
stc = deepcopy(all_stc['same'])
del all_stc

# load source morph
morph = mne.read_source_morph(subs_dir + '0021_LZW/bem/0021_LZW_vol-morph.h5')
stc = morph.apply(stc)
print(stc)

qr = Query(proj_name)
subjects = qr.get_subjects()
subs = range(11,91)
b = 'theta'
times = [0, 4]
if len(argv) > 1:
    b = argv[1]

if len(argv) > 2:
    times = [float(argv[2]), float(argv[3])]

method = 'montecarlo'# FDR
all_data = {}
conds = ['maintenance/mel1','maintenance/mel2','manipulation/mel1','manipulation/mel2']
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        dfname = data_dir + scode + '_TFR_src2_' + b + '.p'
        print('loading file {}'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        dfile.close()
        morph = mne.read_source_morph(subs_dir + scode + '/bem/' + scode + '_vol-morph.h5')
        morph_mat = morph.vol_morph_mat
        if sidx == 0:
            all_data =  {}
        for cd in conds:
            cdata = morph_mat.dot(curdata[cd].crop(times[0],times[1]).data)#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
            print('appending subject {} band {} condition {}'.format(scode,b,cd))
            all_data.setdefault(cd,np.array([cdata]))
            if sidx > 0:
                all_data[cd] = np.vstack((all_data[cd],np.array([cdata])))
    except Exception as e:
        print(e)
        continue

adjacency = mne.spatial_src_adjacency(src_sample)
stats_names = ['all','maintenance','manipulation','maintenance_diff','manipulation_diff','interaction']
conds_math = [['maintenance/mel2','maintenance/mel1','manipulation/mel2','manipulation/mel1'],
              ['maintenance/mel2','maintenance/mel1'],
              ['manipulation/mel2','manipulation/mel1'],
              ['maintenance/mel2','maintenance/mel1'],
              ['manipulation/mel2','manipulation/mel1'],
              ['maintenance/mel2','maintenance/mel1','manipulation/mel2','manipulation/mel1']]
conds_op = [['+','+','+','','4'],['+','','2'],['+','','2'],['-','','1'],['-','','1'],['-','-','+','','1']]

stat_results = {}
for sidx, sn in enumerate(stats_names):
    print('Computing stats for the comparison: ' + sn + '\n\n')
    math_cmd = 'cdata = ('
    for cidx, cd in enumerate(conds_math[sidx]):
        math_cmd += 'all_data["' + cd + '"]' + conds_op[sidx][cidx]
    math_cmd += ')' + '/' + conds_op[sidx][-1]
    print('executing the following command:\n\n' + math_cmd)
    exec(math_cmd)
    print(cdata.shape)
    stat_results[sn] = do_stats(cdata, method=method,adjacency=adjacency,n_permutations=200,n_jobs=-1)

print('\nsaving stats file\n\n')
stats_file = '{}TFR_{}_{}_{}-{}_new.py'.format(stats_dir, b, method, np.round(times[0],2), np.round(times[1],2))
sfile = open(stats_file,'wb')
pickle.dump(stat_results,sfile)
sfile.close()