from stormdb.cluster import ClusterBatch
import os
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
scriptdir = '/projects/{}/scripts/working_memory/'.format(project)
periods = ['encoding']#'encoding','delay']#,'retrieval']
modes = ['patterns','filters']#'patterns','filters']# ['patterns','filters']
cb = ClusterBatch(project)
queue = 'all.q'#'highmem.q'#'all.q'
for p in periods:
    for m in modes:
        cmd = 'python {}APR6n_decoding_patterns_source_stats.py {} {}'.format(scriptdir,p,m)
        cb.add_job(cmd=cmd, queue=queue,n_threads = 12,cleanup = False)
cb.submit()
