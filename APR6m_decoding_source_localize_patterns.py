import mne
import os.path as op
from stormdb.access import Query
from sys import argv
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
mode = 'patterns'
suffix = 'task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05'
qr = Query(proj_name)
subs = qr.get_subjects()
scodes = np.arange(11,91)
modes = ['patterns','filters']
# if len(argv) > 1:
#     scode = int(argv[1])
    
# if len(argv) > 2:
#     mode = argv[2]
for scode in scodes:
    sub = subs[scode-1]
    print('processing subject ', sub)
    for mode in modes:
        try:
            print(mode)
            cfname = op.join(data_dir,sub,sub + '_' + mode + '_' + suffix + '.p')
            times_fname = op.join(data_dir,sub,sub + '_times_' + suffix + '.p')
            inv_fname = op.join(data_dir,sub,sub + '_evoked_inverse_lf_0.05_hf_40_tstep_0.025_twin_0.05.p')

            print('loading data')
            cfile = open(cfname,'rb')
            p = pickle.load(cfile)
            cfile.close()

            times_file = open(times_fname,'rb')
            times = pickle.load(times_file)
            times_file.close()

            inv_file = open(inv_fname,'rb')
            inv = pickle.load(inv_file)
            inv_file.close()

            for cp in p:
                p[cp].times = times[cp[0:-1]]

            p['interaction'] = mne.combine_evoked([p['manipulation1'],p['maintenance1']],weights=[1,-1])

            print('localizing')
            sources = {}
            for e in p:
                sources[e] = mne.beamformer.apply_lcmv(p[e],inv)
                sources[e].tmin = times[cp[0:-1]][0]
                sources[e].tmax = times[cp[0:-1]][-1]
                sources[e].tstep = np.diff(times[cp[0:-1]])[0]

            # Saving
            src_fname = op.join(data_dir,sub,sub + '_{}_sources_{}_localized.p'.format(mode,suffix))
            src_file = open(src_fname,'wb')
            pickle.dump(sources,src_file)
            src_file.close()
        except Exception as e:
            print(e)
            continue