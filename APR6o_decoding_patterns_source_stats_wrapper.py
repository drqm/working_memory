from stormdb.cluster import ClusterBatch
import os
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
scriptdir = '/projects/{}/scripts/working_memory/'.format(project)
#periods = ['all','L1','L2','L3','I1','I2']#,'all','encoding','delay']#'encoding','delay']#,'retrieval']
periods = ['all']
#avg = [1,1,1,1,1,0]#,'False','False','False']
avg = [0]
modes = ['patterns','filters']#'patterns','filters']# ['patterns','filters']
cb = ClusterBatch(project)
queue = 'all.q'#'all.q''all.q'#
for m in modes:    
    for pix,p in enumerate(periods):
        cmd = 'python {}APR6n_decoding_patterns_source_stats.py {} {} {}'.format(scriptdir,p,m,avg[pix])
        cb.add_job(cmd=cmd, queue=queue,n_threads =12,cleanup = False)
cb.submit()

# cmd = 'python {}APR6q_decoding_patterns_source_stats_list_vs_im.py'.format(scriptdir)
# cb.add_job(cmd=cmd, queue=queue,n_threads = 12,cleanup = False)
# cb.submit()