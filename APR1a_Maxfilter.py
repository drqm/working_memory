from os.path import join, basename
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
scratch_folder = join('/projects', proj_name, 'scratch')
#mf_folder = join(scratch_folder, 'maxfilter')
mf_folder = join(scratch_folder, 'maxfiltered_data') # for maxfilter output
scripts_folder = join('/projects', proj_name, 'scripts')
misc_folder = join('/projects', proj_name, 'misc')
trans_folder = join(scratch_folder, 'trans')  # for transforms

from stormdb.access import Query
from stormdb.process import Maxfilter
from mne.io import Raw
from mne.bem import fit_sphere_to_headshape
import warnings
import os
import re
import numpy as np
from sys import argv

from mne.utils import set_log_level
set_log_level('ERROR')

tsss_buffer_len = 16
tsss_corr_lim = 0.96

# if you know that some channels are bad or flat, enter them here
# in the form ['2511', '2241']
static_bad_chans = []

qr = Query(proj_name)
subs = qr.get_subjects()

for ii, ss in enumerate(subs):
    print('{0}: {1}'.format(ii, ss))

cur_sub_ID = 34# see below for the meaning of this
len(subs)
if len(argv) > 1:
    cur_sub_ID = int(argv[1])

cur_sub = subs[cur_sub_ID-1]
print('Current subject: {sub:s}'.format(sub=cur_sub))

sub_specific_bad_chans = []

# Alternatively, just state the ID of the subject
# cur_sub = '0008'

## CALCULATING HEAD POSITIONS
block_names = ['loc','main','inv']
if len(argv) > 2:
    block_names = argv[2:]
#block_names = ['main']

desc_str = ''
for bn in block_names:
    desc_str = desc_str + '*{}*|'.format(bn)
description = (desc_str)

#date_range = ['20210101','20211231']
date_range = None
DATAblocks = qr.filter_series(description=description, subjects=cur_sub,
                              modalities='MEG',study_date_range = date_range)

if len(DATAblocks) != len(block_names):
    raise RuntimeError('Not all blocks found for {0}, please check!'.\
                       format(cur_sub))
for ib in range(len(DATAblocks)):
    print('{:d}: {:s}'.format(ib + 1, DATAblocks[ib]['path']))

info = Raw(os.path.join(DATAblocks[0]['path'], DATAblocks[0]['files'][0]),
           preload=False).info

set_log_level('INFO')
rad, origin_head, ori_dev = fit_sphere_to_headshape(info,
                                                    dig_kinds='extra',
                                                    units='mm')
set_log_level('ERROR')

mf = Maxfilter(proj_name, bad=static_bad_chans + sub_specific_bad_chans)
mfopts = dict(
    origin = '{:.1f} {:.1f} {:.1f}'.format(*tuple(origin_head)),  # mm
    frame = 'head',
    force = True,  # overwrite if needed
    autobad = 'on',  # or use xscan first
    st = True,  # use tSSS
    st_buflen = tsss_buffer_len,  # parameter set in beg. of notebook
    st_corr = tsss_corr_lim,  # parameter set in beg. of notebook
    movecomp = True,#False,#True,
    trans = None,  # compensate to mean initial head position (saved to file),
                              # or use None for initial head position
    logfile = None,  # we replace this in each loop
    hp = None,  # head positions, replace in each loop
    n_threads = 4  # number of parallel threads to run on
    )

out_folder = join(mf_folder,
                  'tsss_st{:d}_corr{:.0f}'.\
                      format(mfopts['st_buflen'],
                             np.round(100 * mfopts['st_corr'])),
                  cur_sub)

# Ensure that output path exists
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
print('Output folder: {:s}'.format(out_folder))
new_desc_str = desc_str.replace('*','')
for blockno, bl in enumerate(DATAblocks):
    for fileno, fil in enumerate(bl['files']):
        in_fname = join(bl['path'], bl['files'][fileno])

        series_name = re.search('({})'.format(new_desc_str),
                                bl['seriename']).group(1)

        out_fname = join(out_folder, '{:s}_raw_tsss.fif'.format(series_name))
        init_pos_fname = os.path.join(DATAblocks[0]['path'], DATAblocks[0]['files'][0])
        if fileno > 0:  # data size > 2 GB
            out_fname = out_fname[:-4] + '-{:d}.fif'.format(fileno)
        mfopts['trans'] = init_pos_fname
        mfopts['logfile'] = out_fname[:-3] + 'log'
        mfopts['hp'] = out_fname[:-3] + 'pos'
        mf.build_cmd(in_fname, out_fname, **mfopts)

# This is not executed, but the line below is
mf.check_input_output_mapping()
mf.print_input_output_mapping()
mf.commands
mf.submit()

try:
    mf_list
except NameError:
    mf_list = []
mf_list += [mf]

for cur_mf in mf_list:
    cur_mf.status
