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

import pickle
from warnings import filterwarnings
import matplotlib.pyplot as plt
filterwarnings("ignore", category=DeprecationWarning)


project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

avg_path = project_dir + '/scratch/invmain_analyses/averages/data/'
qr = Query(project)
sub_codes = qr.get_subjects()

## load data
sub_Ns = [2,11,12,13,14,15,16]#np.arange(8) + 1
gdata = {}
garray = {}
scount = 0

for sub in sub_Ns:
    sub_code = sub_codes[sub-1]
    try:
        print('loading subject {}'.format(sub_code))
        TFR_fname = op.join(avg_path,sub_code + '_TFR.p')
        TFR_file = open(TFR_fname,'rb')
        power = pickle.load(TFR_file)
        TFR_file.close()
        scount = scount +1
        for p in power:
            if scount == 1:
                gdata[p] = []
                garray[p] = []
            power[p].apply_baseline((-2,0))
            gdata[p].append(power[p].data)
            garray[p].append(power[p])
    except:
        print('could not load subject {}'.format(sub_code))
        continue
    
#Grand averages
grand_avg = {}
for e in garray:
    grand_avg[e] = []
    grand_avg[e] = mne.grand_average(garray[e])
    grand_avg[e].data = np.mean(np.array(gdata[e]),0)
    grand_avg[e].comment = garray[e][0].comment

#plots
tfreqs={(0.1,11): (0.1,2),(0.3,2): (0.05,2),(1.2,11): (0.4,2),(2.35,11): (0.2,2),
        (2.4,2): (0.2,2),(4,11): (0.2,2), (5.4,11): (0.4,2)}

tfreqs2={(-1.4,11): (0.5,2), (-1.4,2): (0.05,2),(0.25,11): (0.25,2),(0.25,2): (0.25,2),
        (1,11): (0.25,2),(1,2): (0.25,2), (2,11): (0.25,2),(2,2): (0.25,2), 
        (3,11): (0.25,2), (3,2): (0.25,2), (3.9,11): (0.25,2), (3.9,2): (0.25,2)}   
topomap_args = {'vmax': 5e-22, 'vmin':-5e-22}
topomap_args2 = {'vmax': 3e-22, 'vmin':-3e-22}                            
mfig = grand_avg['main'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),#(-2,0),#None,#(-3,0),
                                                         timefreqs=tfreqs,vmin=-2e-22, vmax=2e-22,
                                                         topomap_args = topomap_args,
                                                         title = 'maintenance block')
mfig.set_figwidth(20)
mfig.set_figheight(10)
#plt.savefig(avg_path + '/figures/{}_TFR_maintenance{}.pdf'.format(sub,suffix),
#            orientation='landscape')

ifig = grand_avg['inv'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),
                                                        timefreqs=tfreqs,vmin=-2e-22, vmax=2e-22,
                                                        topomap_args = topomap_args,
                                                        title = 'manipulation block')
ifig.set_figwidth(20)
ifig.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_manipulation{}.pdf'.format(sub,suffix),
#             orientation='landscape')

dfig = grand_avg['difference'].copy().pick_types(meg = 'grad').plot_joint(baseline=None,
                                                               timefreqs=tfreqs2,vmin=-1e-22, vmax=1e-22,
                                                               topomap_args = topomap_args2,
                                            title = 'difference (manipulation - maintenance)')

dfig.set_figwidth(20)
dfig.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_difference1{}.pdf'.format(sub,suffix),
#             orientation='landscape')

dfig2 = grand_avg['difference'].copy().pick_types(meg = 'grad').plot_joint(baseline=(-1,0),
                                                               timefreqs=tfreqs2,
                                                               topomap_args = topomap_args2,
                                            title = 'difference (manipulation - maintenance)')

dfig2.set_figwidth(20)
dfig2.set_figheight(10)
# plt.savefig(avg_path + '/figures/{}_TFR_difference2{}.pdf'.format(sub,suffix),
#             orientation='landscape')
diff2 = grand_avg['inv']-grand_avg['main']
dfig = grand_avg['difference'].copy().pick_types(meg = 'grad').plot_topo(baseline=(-1,1))
dfig = diff2.copy().pick_types(meg = 'grad').plot_topo(merge_grads =True)

dfig = diff2.copy().pick_types(meg = 'mag').plot_joint(baseline=None,#(-2,0),
                                                               timefreqs=tfreqs2,
                                                               topomap_args = topomap_args2,
                                            title = 'difference (manipulation - maintenance)')
