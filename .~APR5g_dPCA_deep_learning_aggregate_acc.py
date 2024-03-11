import matplotlib.pyplot as plt
import pickle
import numpy as np
from stormdb.access import Query

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/'
acc_dir = wdir + 'scratch/working_memory/deep_learning/data/'
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'

qr = Query(project)
all_subjects = qr.get_subjects()
exclude = [15,32,33,55,58,60,73,76,82]

subjects = [all_subjects[s] for s in range(len(all_subjects)) if (s+1 > 10) & (s+1 < 91) & (s+1 not in exclude)]
periods = ['listen','imagine']

# load data
all_data = []
for s in subjects[:2]:
    cacc = []
    for p in periods:
        f = acc_dir + s + '_accuracy_' + p + '.p'
        with open(f,'rb') as cf:
            cacc += [pickle.load(cf)]
    all_data += [cacc]
    
all_data = np.mean(np.array(all_data)[...,-1],axis=2)

fig = plt.figure()
