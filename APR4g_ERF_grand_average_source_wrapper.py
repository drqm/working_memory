from stormdb.cluster import ClusterBatch
import os
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
scriptdir = '/projects/{}/scripts/working_memory/'.format(project)
periods = ['encoding','delay','retrieval']

cb = ClusterBatch(project)
for p in periods:
    cmd = 'python {}APR4e_ERF_grand_average_sources.py {}'.format(scriptdir,p)
    cb.add_job(cmd=cmd, queue='all.q',n_threads = 6,cleanup = False)
cb.submit()
