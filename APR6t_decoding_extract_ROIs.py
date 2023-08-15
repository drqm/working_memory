proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
scripts_dir = '/projects/' + proj_name + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

import mne
import numpy as np
from stormdb.access import Query
from pickle import load, dump
import os
import os.path as op
from sys import argv

os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'

stats_dir = wdir + 'results/stats/'
#Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()

s =  11
if len(argv) > 1:
    s = int(argv[1])

#suffix = 'patterns_sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized'
#suffix = 'evoked_sources_lf_0.05_hf_40_tstep_0.025_twin_0.05'
suffix = 'evoked_sources_lf_0.05_hf_None_tstep_0.025_twin_0.05'

# Define subject variables 
scode = subjects[s-1]
label_file = subs_dir  + scode + '/mri/aparc+aseg.mgz'
dfname = op.join(data_dir,scode, scode + '_' + suffix + '.p')
src_fname = '{}{}/bem/{}_vol-src.fif'.format(subs_dir,scode,scode)

# load data
print('\n\nloading file {}\n'.format(dfname))
src = mne.read_source_spaces(src_fname)
dfile = open(dfname,'rb')
cdata = load(dfile)
dfile.close()

#### Define ROIs:
ROIs = {}
ROIs['Right Auditory'] = ['ctx-rh-superiortemporal',
                         'ctx-rh-bankssts',
                         'ctx-rh-transversetemporal']
ROIs['Left Auditory'] = ['ctx-lh-superiortemporal',
                         'ctx-lh-bankssts',
                         'ctx-lh-transversetemporal']
ROIs['Right Memory'] = [ 'ctx-rh-fusiform',
                         'ctx-rh-inferiortemporal',
                         'Right-Hippocampus',
                         'ctx-rh-parahippocampal',
                         'ctx-rh-isthmuscingulate',
                         'ctx-rh-precuneus',
                         'ctx-rh-lingual'
                         ]
ROIs['Left Memory'] = ['ctx-lh-fusiform',
                         'ctx-lh-inferiortemporal',
                         'Left-Hippocampus',
                         'ctx-lh-parahippocampal',
                         'ctx-lh-isthmuscingulate',
                         'ctx-lh-precuneus',
                         'ctx-lh-lingual']
ROIs['Right Ventral Cognitive Control'] = [
                         'ctx-rh-medialorbitofrontal',
                         'ctx-rh-lateralorbitofrontal',
                         'Right-Accumbens-area',
                         'Right-Caudate',
                         'Right-Putamen',
                         'Right-Pallidum',  
                         'Right-Thalamus-Proper',
                         'ctx-rh-rostralanteriorcingulate']
ROIs['Left Ventral Cognitive Control'] = [
                         'ctx-lh-medialorbitofrontal',
                         'ctx-lh-lateralorbitofrontal',
                         'Left-Accumbens-area',
                         'Left-Caudate',
                         'Left-Putamen',
                         'Left-Pallidum',  
                         'Left-Thalamus-Proper',
                         'ctx-lh-rostralanteriorcingulate']

ROIs['Right Dorsal Cognitive Control'] = [
                         'ctx-rh-rostralmiddlefrontal',
                         'ctx-rh-superiorfrontal',
                         'ctx-rh-parstriangularis',
                         'ctx-rh-parsopercularis',
                         'ctx-rh-parsorbitalis',
                         'ctx-rh-insula']

ROIs['Left Dorsal Cognitive Control'] = [
                         'ctx-lh-rostralmiddlefrontal',
                         'ctx-lh-superiorfrontal',
                         'ctx-lh-parstriangularis',
                         'ctx-lh-parsopercularis',
                         'ctx-lh-parsorbitalis',
                         'ctx-lh-insula']

# ROIs['Right Mid-Posterior Cingulate'] = [
#                          'ctx_rh_G_and_S_cingul-Mid-Post']
# ROIs['Left Mid-Posterior Cingulate'] = [
#                          'ctx_lh_G_and_S_cingul-Mid-Post']
                       

### Extract ROIs per condition
ROI_data = {}
for c in cdata:
    ROI_data[c] = {'times': cdata[c].times, 'data': {}}
    for ROI in ROIs:
        clabels = ROIs[ROI]
        ROI_data[c]['data'][ROI] = np.mean(cdata[c].extract_label_time_course(labels = [label_file,clabels],
                                                                      src = src,
                                                                      mode = 'auto'),
                                    axis = 0,keepdims=True)

### Save ROI data
out_fname = op.join(data_dir,scode, scode + '_' + suffix + '_ROI.p')
out_file = open(out_fname,'wb')
dump(ROI_data,out_file)
out_file.close()