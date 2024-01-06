import os
from warnings import filterwarnings
from stormdb.cluster import ClusterBatch
from stormdb.access import Query

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

script_dir = '/projects/{}/scripts/working_memory/'.format(project)
qr = Query(project)
subjects = qr.get_subjects()
exclude = [15,32,33,55,58,60,73,76,82]
subjects = [subjects[s] for s in range(len(subjects)) if (s+1 > 10) & (s+1 not in exclude)]
subjects += ['all']
#subjects = ['all']

cb = ClusterBatch(project)

for sub in [subjects[-1]]:
    if sub == 'all':
        csub = [s for s in subjects if s != 'all']
    else:
        csub = [sub]
    submit_cmd = 'python {}APR5a_dPCA_fit.py'.format(script_dir)
    for cs in csub:
        submit_cmd += ' ' + cs
    cb.add_job(cmd=submit_cmd, queue='highmem.q',n_threads = 6, cleanup = False)
cb.submit()
