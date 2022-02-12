# -*- coding: utf-8 -*-
import os
from sys import path
from stormdb.access import Query
from stormdb.cluster import ClusterBatch
from sys import argv

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name
wd = "/projects/{}/".format(proj_name)

ignoreDepWarnings = 0
if ignoreDepWarnings:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
blockList = ['loc','main','inv']
if len(argv) > 2:
    blockList = argv[2:]
sub_codes = [34,35]#,26,27,28,29,30,31]#[20,21,22,23,24,25,26,27,28,29,30,31] #9,10]
if len(argv) > 1:
    sub_codes = [int(argv[1])]
qr = Query(proj_name)
subjects = qr.get_subjects()
cb = ClusterBatch(proj_name)
for sub_code in sub_codes:
    subj= subjects[sub_code-1]
    raw_dir = '{}scratch/maxfiltered_data/tsss_st16_corr96/{}/'.format(wd,subj)
    ICA_dir= '{}scratch/working_memory/ICA/{}/'.format(wd,subj)

    if not(os.path.exists(ICA_dir)):
        os.mkdir(ICA_dir)
        print('*** Directory created')

    for bname in blockList:
        in_fname = raw_dir + bname + '_raw_tsss.fif'
        out_fname = ICA_dir + bname + '_raw_tsss-ica.fif'
        submit_cmd = 'python APR2a_runICA_cluster.py {} {} {} {} {}'.format(in_fname,
                                                                         out_fname,
                                                                         ICA_dir,
                                                                         bname,
                                                                         subj)
        cb.add_job(cmd=submit_cmd, queue='highmem.q',n_threads = 2,cleanup = False)#,total_memory='16G')
cb.submit()
