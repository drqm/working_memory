import mne
#%gui qt
#import matplotlib
#%matplotlib qt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats
from mne.datasets import sample
from mne.stats import spatio_temporal_cluster_1samp_test
import os
from os import path as op
import pickle
from copy import deepcopy
import warnings
from do_stats import do_stats
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# os.environ['ETS_TOOLKIT'] = 'qt4'
# os.environ['QT_API'] = 'pyqt5'
# %gui qt
#mne.viz.set_3d_backend("notebook")


project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']= '~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

avg_path = project_dir + '/scratch/working_memory/averages/data/'
stats_dir = project_dir + '/scratch/working_memory/results/stats/'
figures_dir = project_dir + '/scratch/working_memory/results/figures/'

dem_file = project_dir + '/misc/WM_demographics.csv'
qr = Query(project)
sub_codes = qr.get_subjects()

## Select task:
loc_source = True
loc_sensor = False
seq_sensor = False

## Load data:
sub_Ns = np.arange(11,91) 
#exclude = np.array([55,60,73,82]) # subjects with low maintenance accuracy
sdata = {}
scodes = []
scount = 0
cshape = {}
for sub in sub_Ns:
    sub_code = sub_codes[sub-1]
    #if sub not in exclude:
    try:
        print('loading subject {}'.format(sub_code))
        if loc_source:
            score_fname = op.join(avg_path,sub_code + '_scores_loc_source.p')
        if loc_sensor:
            score_fname = op.join(avg_path,sub_code + '_scores_loc_sensor.p')
        if seq_sensor:
            score_fname = op.join(avg_path,sub_code + '_scores_sensor_seq_imagine_seq.p')
        score_file = open(score_fname,'rb')
        score = pickle.load(score_file)
        score_file.close()
        scodes.append([sub])
        scount = scount + 1
        for c in score:              
            if scount == 1:
                sdata[c] = []
                cshape[c] = score[c].data.shape[0]
            if score[c].shape[0] == cshape[c]:
                sdata[c].append(score[c].data)
    except Exception as e:
        print('could not load subject {}'.format(sub_code))
        print(e)
        continue 
        
print(len(scodes))

## Load demographics:
dem = pd.read_csv(dem_file)
dem = dem[np.isin(dem['Subject'],scodes)]
print(dem)

## Grand averages:
smean, sstd, smedian, siqr_lower, siqr_upper = {},{},{},{},{}
for s in sdata:
    sdata[s] = np.array(sdata[s])
    print(sdata[s].shape)
    smean[s] = np.mean(sdata[s],0)
    smedian[s] = np.median(sdata[s],0)
    sstd[s] = np.std(sdata[s],0)
    siqr_lower[s] = np.percentile(sdata[s],25,0)
    siqr_upper[s] = np.percentile(sdata[s],75,0)
    
ncols = 2 #2 columns and 2 rows for loc #4 columns and 4 rows for seq
fig, axes = plt.subplots(ncols=ncols,nrows=2, figsize = (20,13)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(smean):
    f,se = s.split('_from_')
    ext = [-.25,smean[s].shape[1] * .01 - .01, #.025 - 0.25,  
           -.25,smean[s].shape[0] * .01 - .01] #.025 - 0.25]
    rix, cix = sidx//ncols,sidx%ncols
    im = axes[rix, cix].matshow(smean[s], vmin = .23, vmax = .43,#vmin=0.18, vmax=0.48,
                                      cmap='RdBu_r', origin='lower', extent=ext)
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    axes[rix, cix].xaxis.set_ticks_position('bottom')
    axes[rix, cix].set_xlabel('Testing Time (s)')
    axes[rix, cix].set_ylabel('Training Time (s)')
    axes[rix, cix].set_anchor('W')
    axes[rix, cix].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.suptitle('Decoding accuracy (ROC - AUC)', fontsize =  20)
plt.tight_layout()
if loc_source:
    plt.savefig(figures_dir + 'accuracies_imagined_uncorrected_loc_source.pdf',orientation='landscape')
if loc_sensor:
    plt.savefig(figures_dir + 'accuracies_imagined_uncorrected_loc_sensor.pdf',orientation='landscape')
if seq_sensor:
    plt.savefig(figures_dir + 'accuracies_imagined_uncorrected_seq_sensor.pdf',orientation='landscape')

## False Discovery Rate correction:
FDR_stats = {}
for s in sdata:
    FDR_stats[s] = do_stats(sdata[s],'FDR',h0 = .33) #.5)

ncols = 2 #2 columns and 2 rows for loc #4 columns and 4 rows for seq
fig, axes = plt.subplots(ncols=ncols,nrows=2, figsize = (20,13)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(FDR_stats):
    f,se = s.split('_from_')
    ext = [-.25,FDR_stats[s]['data_mean'].shape[0] * .01 - .01, #.025 - 0.25,  
           -.25,FDR_stats[s]['data_mean'].shape[1] * .01 - .01] #.025 - 0.25]
    rix, cix = sidx//ncols,sidx%ncols
    #mask = FDR_stats[s]['qvals'] <= .025
    mask = FDR_stats[s]['mask']
    im = axes[rix, cix].matshow(FDR_stats[s]['data_mean'].T * mask.T, vmin = .23, vmax = .43,#vmin=0.18, vmax=0.48,
                                      cmap='RdBu_r', origin='lower', extent=ext)
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    axes[rix, cix].xaxis.set_ticks_position('bottom')
    axes[rix, cix].set_xlabel('Testing Time (s)')
    axes[rix, cix].set_ylabel('Training Time (s)')
    axes[rix, cix].set_anchor('W')
    axes[rix, cix].set_title('pred. {} from {}'.format(f,se),{'horizontalalignment': 'center'})
cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.suptitle('Decoding accuracy (ROC - AUC)', fontsize =  20)
plt.tight_layout()
if loc_source:
    plt.savefig(figures_dir + 'accuracies_imagined_FDR_loc_source.pdf',orientation='landscape')
if loc_sensor:
    plt.savefig(figures_dir + 'accuracies_imagined_FDR_loc_sensor.pdf',orientation='landscape')
if seq_sensor:
    plt.savefig(figures_dir + 'accuracies_imagined_FDR_seq_sensor.pdf',orientation='landscape')

## Cluster correction:
cluster_stats = {}
for s in sdata:
    print('doing stats for {}'.format(s))
    cluster_stats[s] = do_stats(sdata[s],method='montecarlo',h0=.33,n_permutations=5000)
                                
ncols = 2 #2 columns and 2 rows for loc #4 columns and 4 rows for seq
fig, axes = plt.subplots(ncols=ncols,nrows=2, figsize = (20,13)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(cluster_stats):
    f,se = s.split('_from_')
    ext = [-.25,cluster_stats[s]['data_mean'].shape[1] * .01 - .01, #.025 - 0.25,  
           -.25,cluster_stats[s]['data_mean'].shape[0] * .01 - .01] #.025 - 0.25]
    rix, cix = sidx//ncols,sidx%ncols
    #mask = stats_results[s]['qvals'] <= .025
    mask = cluster_stats[s]['mask']
    im = axes[rix, cix].matshow(cluster_stats[s]['data_mean'] * mask, vmin = .23, vmax = .43,#vmin=0.18, vmax=0.48,
                                      cmap='RdBu_r', origin='lower', extent=ext)
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    axes[rix, cix].xaxis.set_ticks_position('bottom')
    axes[rix, cix].set_xlabel('Testing Time (s)')
    axes[rix, cix].set_ylabel('Training Time (s)')
    axes[rix, cix].set_anchor('W')
    axes[rix, cix].set_title('pred. {} from {}'.format(f,se),{'horizontalalignment': 'center'})
cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.suptitle('Decoding accuracy (ROC - AUC)', fontsize =  20)
plt.tight_layout()
if loc_source:
    plt.savefig(figures_dir + 'accuracies_imagined_cluster_loc_source.pdf',orientation='landscape')
if loc_sensor:
    plt.savefig(figures_dir + 'accuracies_imagined_cluster_loc_sensor.pdf',orientation='landscape')
if seq_sensor:
    plt.savefig(figures_dir + 'accuracies_imagined_cluster_seq_sensor.pdf',orientation='landscape')

## Linear regression:
betas, rval, reg_pvals, ci, aic = {},{},{},{},{}
preds = np.array([dem['vividness'],dem['musicianship']]).T
preds = sm.add_constant(preds)
print(preds.shape)
for s in sdata:
    cdata = np.array(sdata[s].copy()) - 0.5
    xshape = cdata.shape[1]
    yshape = cdata.shape[2]
    betas[s] = np.zeros((preds.shape[1],xshape,yshape))
    rval[s] = np.zeros((xshape,yshape))
    aic[s] = np.zeros((xshape,yshape))
    reg_pvals[s] = np.zeros((preds.shape[1],xshape,yshape))
    #ci[s] = np.zeros(betas[s].shape)
    for y in range(yshape):
        for x in range(xshape):
            print('regressing condition {} training sample: {}/{}, testing sample: {}/{}'.format(s,x,xshape,y,yshape))
            res = sm.OLS(cdata[:,x,y],preds,missing='drop').fit()
            betas[s][:,x,y] = res.params
            reg_pvals[s][:,x,y] = res.pvalues
            rval[s][x,y] = res.rsquared
            aic[s][x,y] = res.aic

## Correlations with vividness:
pvals = {}
cors = {}
for s in sdata:
    sdata[s] = np.array(sdata[s])
    xshape = sdata[s].shape[1]
    yshape = sdata[s].shape[2]
    cors[s] = np.zeros((xshape,yshape))
    pvals[s] = np.zeros((xshape,yshape))
    for y in range(yshape):
        for x in range(xshape):
            print('correlating condition {} training sample: {}/{}, testing sample: {}/{}'.format(s,x,xshape,y,yshape))
            nanix = np.isnan(np.array(dem['vividness'])) == False
            #print(nanix)
#             cur_cor = np.array(pd.DataFrame({'a': sdata[s][:,x,y], 'b': np.array(dem['vividness'])}).corr())[0,1]
#             print(cur_cor)
            cors[s][x,y], pvals[s][x,y] = stats.pearsonr(sdata[s][nanix,x,y], np.array(dem['vividness'][nanix]))
            
ncols = 4
fig, axes = plt.subplots(ncols=ncols,nrows=4, figsize = (20,13)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(cors):
    f,se = s.split('_from_')
    ext = [-.25,smean[s].shape[1] * .025 - 0.25,  
           -.25,smean[s].shape[0] * .025 - 0.25]
    rix, cix = sidx//ncols,sidx%ncols
    mask = pvals[s] <= .025
    im = axes[rix, cix].matshow(cors[s]*mask, vmin = -1, vmax = 1,#vmin=0.18, vmax=0.48,
                                      cmap='RdBu_r', origin='lower', extent=ext)
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    axes[rix, cix].xaxis.set_ticks_position('bottom')
    axes[rix, cix].set_xlabel('Testing Time (s)')
    axes[rix, cix].set_ylabel('Training Time (s)')
    axes[rix, cix].set_anchor('W')
    axes[rix, cix].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.suptitle('Vividness correlation', fontsize =  20)
plt.tight_layout()

# Plot musicians and nonmusicians:
varnames = {'intercept': 0, 'vividness': 1, 'musicianship': 2}
ncols = 4
for ex in varnames:
    fig, axes = plt.subplots(ncols=ncols,nrows=4, figsize = (20,13)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
    for sidx,s in enumerate(betas):
        f,se = s.split('_from_')
        ext = [-.25,smean[s].shape[1] * .025 - 0.25,  
               -.25,smean[s].shape[0] * .025 - 0.25]
        rix, cix = sidx//ncols,sidx%ncols
        mask = reg_pvals[s][varnames[ex]] < .025
        im = axes[rix, cix].matshow(betas[s][varnames[ex]]*mask,# vmin = .4, vmax = .6,#vmin=0.18, vmax=0.48,
                                          cmap='RdBu_r', origin='lower', extent=ext)
        axes[rix, cix].axhline(0., color='k')
        axes[rix, cix].axvline(0., color='k')
        axes[rix, cix].xaxis.set_ticks_position('bottom')
        axes[rix, cix].set_xlabel('Testing Time (s)')
        axes[rix, cix].set_ylabel('Training Time (s)')
        axes[rix, cix].set_anchor('W')
        axes[rix, cix].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
    cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
    fig.colorbar(im,cax=cbar_ax)
    fig.suptitle('Decoding accuracy (ROC - AUC) {}'.format(ex), fontsize =  20)
    plt.tight_layout()
#plt.savefig(avg_path + '/figures/{}_accuracies_imagined.pdf'.format(sub),orientation='landscape')

eix0 = np.array(dem['musicianship']) == 0
eix1 = np.array(dem['musicianship']) == 1
print(eix0)
print(eix1)
fig, axes = plt.subplots(ncols=ncols,nrows=4, figsize = (20,13)) #,gridspec_kw=dict(width_ratios=[1,1,1,1]) )
for sidx,s in enumerate(sdata):
    f,se = s.split('_from_')

    ext = [-.25,smean[s].shape[1] * .025 - 0.25,  
           -.25,smean[s].shape[0] * .025 - 0.25]
    rix, cix = sidx//ncols,sidx%ncols
    im = axes[rix, cix].matshow(sdata[s][eix1].mean(axis=0) - sdata[s][eix0].mean(axis=0), #vmin = .4, vmax = .6,#vmin=0.18, vmax=0.48,
                                      cmap='RdBu_r', origin='lower', extent=ext)
    axes[rix, cix].axhline(0., color='k')
    axes[rix, cix].axvline(0., color='k')
    axes[rix, cix].xaxis.set_ticks_position('bottom')
    axes[rix, cix].set_xlabel('Testing Time (s)')
    axes[rix, cix].set_ylabel('Training Time (s)')
    axes[rix, cix].set_anchor('W')
    axes[rix, cix].set_title('pred. {}'.format(s),{'horizontalalignment': 'center'})
cbar_ax = fig.add_axes([0.925,0.15,0.01,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.suptitle('Decoding accuracy (ROC - AUC) musicians - non-musicians', fontsize =  20)
plt.tight_layout()


