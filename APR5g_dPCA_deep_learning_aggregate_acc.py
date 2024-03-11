import matplotlib.pyplot as plt
import pickle
import numpy as np
from stormdb.access import Query

wdir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/'
acc_dir = wdir + 'scratch/working_memory/deep_learning/data/'
fig_dir = wdir + 'scratch/working_memory/deep_learning/figures/'
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'

qr = Query(project)
all_subjects = qr.get_subjects()
exclude = [15,32,33,55,58,60,73,76,82]

subjects = [all_subjects[s] for s in range(len(all_subjects)) if (s+1 > 10) & (s+1 < 91) & (s+1 not in exclude)]
periods = ['listen','imagine']

# load data
all_data = []
for s in subjects:
    cacc = []
    for p in periods:
        f = acc_dir + s + '_accuracy_' + p + '.p'
        with open(f,'rb') as cf:
            cacc += [pickle.load(cf)]
    all_data += [cacc]
    
all_data = np.mean(np.array(all_data)[...,-10:],axis=(2,3))

n = all_data.shape[0]

## Plot accuracies
fig, ax = plt.subplots()
ax.boxplot(all_data*100, widths = 0.05,showfliers=False,positions=[1.075,2.075])
vp = ax.violinplot(all_data*100, showmeans=False, points=1000,showextrema=False, showmedians=False)#, vert=False)
for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color('b')
    b.get_paths()[0].vertices[:, 0] += 0.075 

x1 = 1+np.random.uniform(-.01,0.01,all_data.shape[0])
x2 = 2+np.random.uniform(-.01,0.01,all_data.shape[0])
ax.plot(x1, all_data[:,0]*100, '.',color='k',alpha=.25,markersize=9)
ax.plot(x2, all_data[:,1]*100,'.',color='k',alpha=.25,markersize=9)
ax.set_ylim([0,100])
ax.set_ylabel('accuracy (% correct)')
ax.axhline(25,color='k')
for chl in np.arange(0,100,10):
    ax.axhline(chl,color='k',alpha=.1,linewidth=0.5)
ax.set_xticklabels(periods)
plt.savefig(fig_dir + 'accuracy.pdf')



