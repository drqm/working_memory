#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
from scipy import stats
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
from do_stats import do_stats
filterwarnings("ignore", category=DeprecationWarning)

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']= '~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

avg_path = project_dir + '/scratch/working_memory/averages/data/'
stats_dir = project_dir + '/scratch/working_memory/results/stats/'
qr = Query(project)
sub_codes = qr.get_subjects()

## load data
sub_Ns = np.arange(11,91) #[2,11,12,13,14,15,16]#np.arange(8) + 1
#exclude = np.array([55,60,73,82]) # subjects with low maintenance accuracy
gdata = {}
garray = {}
scount = 0
for sub in sub_Ns:
    sub_code = sub_codes[sub-1]
    try:
        print('loading subject {}'.format(sub_code))
        evkd_fname = op.join(avg_path,sub_code + '_patterns_loc_sensor.p')
        evkd_file = open(avg_path + sub_code + '_patterns_loc_sensor.p','rb')
        evokeds = pickle.load(evkd_file)
        evkd_file.close()
        scount = scount +1
        for e in evokeds:
            for tone in evokeds[e]:
                condname = e + '_' + tone
                if scount == 1:
                    gdata[condname] = []
                    garray[condname] = []
                gdata[condname].append(evokeds[e][tone].data)
                garray[condname].append(evokeds[e][tone])
    except Exception as ex:
        print('could not load subject {}'.format(sub_code))
        print(ex)
        continue
for g in gdata:
    gdata[g] = np.array(gdata[g])

grand_avg = {}
for e in garray:
    grand_avg[e] = mne.grand_average(garray[e])
    grand_avg[e].data = np.mean(np.array(gdata[e]),0)
    grand_avg[e].comment = garray[e][0].comment

stats_results = {}
for e in gdata:
    stats_results[e] = do_stats(gdata[e], method='FDR')


