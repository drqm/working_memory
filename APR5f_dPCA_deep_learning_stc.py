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
all_subjects = qr.get_subjects()
exclude = [15,32,33,55,58,60,73,76,82]

subjects = [all_subjects[s] for s in range(len(all_subjects)) if (s+1 > 10) & (s+1 < 91) & (s+1 not in exclude)]
periods = ['baseline']#,'imagine','listen','baseline']
#subjects = ['all']

cb = ClusterBatch(project)

for sub in subjects:
  # for f in ['False','True']:
  #   for n in [2,4]:
    for p in periods:
        print(sub,p)# f, n
        submit_cmd = f'python {script_dir}APR5e_dPCA_deep_learning.py {sub} {p}'# {n} {f}
        os.system(submit_cmd)

#         cb.add_job(cmd=submit_cmd, queue='all.q', cleanup = False)#n_threads=4
# cb.submit()
