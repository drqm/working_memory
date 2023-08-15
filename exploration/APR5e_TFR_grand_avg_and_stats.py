#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mne
import os
import os.path as op
import numpy as np
import pickle
from scipy import stats
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
filterwarnings("ignore", category=DeprecationWarning)

import pickle
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)


project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne3d' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

avg_path = project_dir + '/scratch/working_memory/averages/data/'
qr = Query(project)
sub_codes = qr.get_subjects()

## load data
sub_Ns = np.arange(21,90)
gdata = {}
garray = {}
scount = 0
conds = ['difference']
for sub in sub_Ns:
    sub_code = sub_codes[sub-1]
    print('loading subject {}'.format(sub_code))
    try:
        TFR_fname = op.join(avg_path,sub_code + '_TFRtw_2-4.p')
        TFR_file = open(TFR_fname,'rb')
        power = pickle.load(TFR_file)
        TFR_file.close()
        scount = scount + 1 #+12,11,12,13,14,15,16
        for p in conds:#power:
            cdata = power[p].data#.reshape((1, power[p].data.shape[0],
#                                              power[p].data.shape[1],
#                                              power[p].data.shape[2]))
            if scount == 1:
                gdata[p] = []#cdata
#            #else:
            gdata[p].append(cdata)#np.append(gdata[p], cdata,axis=0)
#                #garray[p] = []
#            #power[p].apply_baseline((-2,0))
#            #garray[p].append(power[p])
#            #garray[p][-1].data = []
#        #el(power)
    except:
        print('could not load subject {}'.format(sub_code))
        continue

#Grand averages
grand_avg = {}
for e in gdata:
    grand_avg[e] = power[e].copy()
#    #grand_avg[e] = mne.grand_average(garray[e])
    gdata[e] = np.array(gdata[e])
    grand_avg[e].data = np.mean(gdata[e],0)
#    #grand_avg[e].comment = garray[e][0].comment

#del gdata

#grand_avg['difference'].plot_topo()

#stats
grad_ix = np.sort(np.concatenate((np.arange(1,306,3),np.arange(2,306,3))))
mag_ix = np.arange(0,306,3)
# out = stats.ttest_1samp(gdata['difference'],0,axis=0)
# qvals = mne.stats.fdr_correction(out[1])
# dummy = grand_avg['difference'].copy()
# dummy.data = dummy.data*qvals[0]
# dummy.plot_topo()

cdata = gdata['difference'].transpose([0,3,2,1])
#cdata = cdata[:,:,:,grad_ix]
adj = mne.channels.find_ch_adjacency(power['main'].info, 'grad')
adj = mne.stats.combine_adjacency(adj[0], cdata.shape[1], cdata.shape[2])
cstats = mne.stats.permutation_cluster_1samp_test(cdata[:,:,:,grad_ix],
                                                  n_permutations=100, adjacency = adj)
np.min(cstats[2])

#plots
# tfreqs={(0.1,11): (0.1,2),(0.3,2): (0.05,2),(1.2,11): (0.4,2),(2.35,11): (0.2,2),
#         (2.4,2): (0.2,2),(4,11): (0.2,2), (5.4,11): (0.4,2)}
#
# tfreqs2={(-1.4,11): (0.5,2), (-1.4,2): (0.05,2),(0.25,11): (0.25,2),(0.25,2): (0.25,2),
#         (1,11): (0.25,2),(1,2): (0.25,2), (2,11): (0.25,2),(2,2): (0.25,2),
#         (3,11): (0.25,2), (3,2): (0.25,2), (3.9,11): (0.25,2), (3.9,2): (0.25,2)}
# topomap_args = {'vmax': 5e-22, 'vmin':-5e-22}
# topomap_args2 = {'vmax': 3e-22, 'vmin':-3e-22}
# mfig = grand_avg['main'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),#(-2,0),#None,#(-3,0),
#                                                          timefreqs=tfreqs,vmin=-2e-22, vmax=2e-22,
#                                                          topomap_args = topomap_args,
#                                                          title = 'maintenance block')
# mfig.set_figwidth(20)
# mfig.set_figheight(10)
# #plt.savefig(avg_path + '/figures/{}_TFR_maintenance{}.pdf'.format(sub,suffix),
# #            orientation='landscape')
#
# ifig = grand_avg['inv'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),
#                                                         timefreqs=tfreqs,vmin=-2e-22, vmax=2e-22,
#                                                         topomap_args = topomap_args,
#                                                         title = 'manipulation block')
# ifig.set_figwidth(20)
# ifig.set_figheight(10)
# # plt.savefig(avg_path + '/figures/{}_TFR_manipulation{}.pdf'.format(sub,suffix),
# #             orientation='landscape')
#
# dfig = grand_avg['difference'].copy().pick_types(meg = 'grad').plot_joint(baseline=None,
#                                                                timefreqs=tfreqs2,vmin=-1e-22, vmax=1e-22,
#                                                                topomap_args = topomap_args2,
#                                             title = 'difference (manipulation - maintenance)')
#
# dfig.set_figwidth(20)
# dfig.set_figheight(10)
# # plt.savefig(avg_path + '/figures/{}_TFR_difference1{}.pdf'.format(sub,suffix),
# #             orientation='landscape')
#
# dfig2 = grand_avg['difference'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),
#                                                                timefreqs=tfreqs2,
#                                                                topomap_args = topomap_args2,
#                                             title = 'difference (manipulation - maintenance)')
#
# dfig2.set_figwidth(20)
# dfig2.set_figheight(10)
# # plt.savefig(avg_path + '/figures/{}_TFR_difference2{}.pdf'.format(sub,suffix),
# #             orientation='landscape')
# diff2 = grand_avg['inv']-grand_avg['main']
# dfig = grand_avg['difference'].copy().pick_types(meg = 'grad').plot_topo(baseline=(-1,1))
# dfig = diff2.copy().pick_types(meg = 'grad').plot_topo(merge_grads =True)
#
# dfig = diff2.copy().pick_types(meg = 'mag').plot_joint(baseline=None,#(-2,0),
#                                                                timefreqs=tfreqs2,
#                                                                topomap_args = topomap_args2,
#                                             title = 'difference (manipulation - maintenance)')
