#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
filterwarnings("ignore", category=DeprecationWarning)

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
#os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

avg_path = project_dir + '/scratch/working_memory/averages/data/'
qr = Query(project)
sub_codes = qr.get_subjects()
# sub_codes = ['0001_VLC','0002_BYG','0003_S5V','0004_LXY','0005_AO8','0006_TNV',
#              '0007_ESO','0008_HMD','0009_XJB','0010_RKU','0011_U7X','0012_VK2']

## load data
sub_Ns = [2,11,12,13,14,15,16]#np.arange(8) + 1
gdata = {}
garray = {}
scount = 0
for sub in sub_Ns:
    sub_code = sub_codes[sub-1]
    try:
        print('loading subject {}'.format(sub_code))
        evkd_fname = op.join(avg_path,sub_code + '_evoked.p')
        evkd_file = open(avg_path + sub_code + '_evoked.p','rb')
        evokeds = pickle.load(evkd_file)
        evkd_file.close()
        scount = scount +1
        for e in evokeds:
            if scount == 1:
                gdata[e] = {}
                garray[e] = {}
            for c in evokeds[e]:
                if scount == 1:
                    gdata[e][c] = []
                    garray[e][c] = []
                gdata[e][c].append(evokeds[e][c].data)
                garray[e][c].append(evokeds[e][c])
    except:
        print('could not load subject {}'.format(sub_code))
        continue

# Grand averages
grand_avg = {}
for e in garray:
    grand_avg[e] = {}
    for c in garray[e]:
        grand_avg[e][c] = mne.grand_average(garray[e][c])
        grand_avg[e][c].data = np.mean(np.array(gdata[e][c]),0)
        grand_avg[e][c].comment = garray[e][c][0].comment

mne.viz.plot_evoked_topo([grand_avg['main']['all'].copy().pick_types('mag').crop(-0.5,2),
                          grand_avg['inv']['all'].copy().pick_types('mag').crop(-0.5,2)])

mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('mag').crop(-0.5,2))
grand_avg['difference']['all'].copy().pick_types('grad').crop(-0.5,2).plot_joint()

mne.viz.plot_evoked_topo([grand_avg['main']['all'].copy().pick_types('mag').crop(2,4.5),
                          grand_avg['inv']['all'].copy().pick_types('mag').crop(2,4.5)])
mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('grad').crop(2,4.5))
grand_avg['difference']['all'].copy().pick_types('grad').crop(2,4.5).plot_joint()

mne.viz.plot_evoked_topo([grand_avg['main']['different1'].copy().pick_types('mag').crop(4,6.5),
                          grand_avg['main']['different2'].copy().pick_types('mag').crop(4,6.5)])
mne.viz.plot_evoked_topo([grand_avg['inv']['other1'].copy().pick_types('mag').crop(4,6.5),
                          grand_avg['inv']['other2'].copy().pick_types('mag').crop(4,6.5)])

mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('grad').crop(4,6.5))
grand_avg['difference']['all'].copy().pick_types('grad').crop(2,4.5).plot_joint()

mne.viz.plot_evoked_topo([grand_avg['main']['all'].copy().pick_types('grad'),
                          grand_avg['inv']['all'].copy().pick_types('grad')],
                          merge_grads=True)
mne.viz.plot_evoked_topo(grand_avg['difference']['all'].copy().pick_types('grad'),
                         merge_grad=True)

grand_avg['difference']['all'].copy().pick_types('grad').crop(-0.5,2).plot_joint

# Encoding plots:
chans = ['MEG1341','MEG0311','MEG2241']
for ch in chans:
    mne.viz.plot_compare_evokeds([grand_avg['main']['all'].copy().crop(-0.5,2),
                                  grand_avg['inv']['all'].copy().crop(-0.5,2)],
                                  picks=ch)

grand_avg['difference']['all'].copy().pick_types('mag').plot_topomap(times=[0.25,0.5,0.75,1])
grand_avg['difference']['all'].copy().pick_types('grad').plot_topomap(times=[0.25,0.5,0.75,1])
grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=[0.125,0.21,0.45])
grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=[0.125,0.21,0.45])

# Mainteance/manipulation plots:
chans = ['MEG1341','MEG0311','MEG2241']
for ch in chans:
    mne.viz.plot_compare_evokeds([grand_avg['main']['all'].copy().crop(1.5,4.5),
                                  grand_avg['inv']['all'].copy().crop(1.5,4.5)],
                                  picks=ch)
grand_avg['difference']['all'].copy().pick_types('mag').plot_topomap(times=np.arange(3,4.25,0.25),
                                                                     average=0.25)
grand_avg['difference']['all'].copy().pick_types('grad').plot_topomap(times=np.arange(3,4.25,0.25),
                                                                     average=0.25)
grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=3.6)
grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=3.6)

# retrieval plots:
chans = ['MEG1341','MEG0311','MEG2241']
for ch in chans:
    mne.viz.plot_compare_evokeds([grand_avg['main']['all'].copy().crop(4,6.5),
                                  grand_avg['inv']['all'].copy().crop(4,6.5)],
                                  picks=ch)
grand_avg['difference']['all'].copy().pick_types('mag').plot_topomap(times=[4.75,5,5.25,5.5,5.75])
grand_avg['difference']['all'].copy().pick_types('grad').plot_topomap(times=[4.75,5,5.25,5.5,5.75])
grand_avg['main']['all'].copy().pick_types('mag').plot_topomap(times=5.4)
grand_avg['main']['all'].copy().pick_types('grad').plot_topomap(times=5.4)

# Dev type plots:
chans = ['MEG0431']#,'MEG0311','MEG2241']
for ch in chans:
    mne.viz.plot_compare_evokeds([grand_avg['main']['same'].copy().crop(4,6.5),
                                  grand_avg['main']['different1'].copy().crop(4,6.5),
                                  grand_avg['main']['different2'].copy().crop(4,6.5)],
                                  picks=ch)
    mne.viz.plot_compare_evokeds([grand_avg['inv']['inverted'].copy().crop(4,6.5),
                              grand_avg['inv']['other1'].copy().crop(4,6.5),
                              grand_avg['inv']['other2'].copy().crop(4,6.5)],
                              picks=ch)

## compute group sources:

# Download fsaverage files


# The files live in:
# subjects_dir = op.join(project_dir,'scratch/fs_subjects_dir')
# inv_file = open(avg_path + '0002_BYG_evoked_inverse.p','rb')
# inv = pickle.load(inv_file)
# inv_file.close()

# src_file = open(avg_path + '0002_BYG_evoked_sources.p','rb')
# sources = pickle.load(src_file)
# src_file.close()

# SNR = 3
# src_diff = sources['inv']['all']-sources['main']['all']
# sources['main']['all'].plot(subjects_dir=subjects_dir,initial_time=0.11,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])
# sources['inv']['all'].plot(subjects_dir=subjects_dir,initial_time=0.11,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])

# src_diff.plot(subjects_dir=subjects_dir,initial_time=0.11,hemi = 'split',
#               time_viewer=True, views=['lateral','medial'])



# src_diff2 = sources['main']['different2']-sources['main']['different1']
# src_diff3 = sources['inv']['other2']-sources['inv']['other1']
# src_diff2.plot(subjects_dir=subjects_dir,initial_time=5.22,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])
# src_diff3.plot(subjects_dir=subjects_dir,initial_time=5.22,hemi = 'split',
#                             time_viewer=True, views=['lateral','medial'])
