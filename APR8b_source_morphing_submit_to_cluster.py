# -*- coding: utf-8 -*-
import os
import sys
from sys import path
from stormdb.access import Query
from stormdb.cluster import ClusterBatch

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']= proj_name
wd = "/projects/{}/".format(proj_name)
#subs_path = wd + 'scratch/main_analyses/bads_table.csv'
path.append('{}scripts/working_memory'.format(wd))

ignoreDepWarnings = 0
if ignoreDepWarnings:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# list subject codes
# default subject
sub_codes = range(11,91)

#sub_codes = [11,12,13,14,15,16,17,18,19,20]#,41,42,44,45,46,47,48,49,50,51]
# change if argument given
if len(sys.argv) > 1:
    sub_codes = [int(sys.argv[1])]

qr = Query(proj_name)
subjects = qr.get_subjects()
cb = ClusterBatch(proj_name)
for sub_code in sub_codes:
    #subj= subjects[sub_code-1]
    submit_cmd = 'python APR8a_source_morphing.py {}'.format(sub_code)
    cb.add_job(cmd=submit_cmd, queue='all.q',n_threads = 2,cleanup = False)
cb.submit()
