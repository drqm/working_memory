from stormdb.cluster import ClusterBatch
import os
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
scriptdir = '/projects/{}/scripts/working_memory/'.format(project)
bands = ['delta','theta','alpha','beta1','beta2','HFA']
times = [[0,2],[2,4],[4, 6.25]]
#times = [[4,6.25]]
cb = ClusterBatch(project)
for b in bands:
    for t in times:
        cmd = 'python {}APR5h_TFR_grand_average_sources.py {} {} {}'.format(scriptdir,b,t[0],t[1])
        cb.add_job(cmd=cmd, queue='all.q',n_threads = 8,cleanup = False)
cb.submit()
