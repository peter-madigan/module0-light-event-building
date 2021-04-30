#!/usr/bin/env python
'''
Light system ADC event builder

Generates an hdf5 file representing light system events, to be used for further aggregation
and analysis.

Performs a simple event building process based on the timestamps of light data and a low-level
reconstruction of prompt/delayed signals

Output data structure is:

light_event - event [u8] (N,): event indexing / identifier
            - sn [u8] (N, n_adcs): serial number for each adc in event
            - ch [u1] (N, n_adcs, n_channels): channel id for each adc channel
            - utime_ms [u1] (N, n_adcs): timestamp in ms
            - tai_ns [u8] (N, n_adcs): clock timestamp in ns
            - wvfm_valid [u1] (N, n_adcs, n_channels): 1 if channel present in event
light_wvfm  - samples [i2] (N, n_adcs, n_channels, n_samples): raw waveforms of each channel in event
light_reco  - ped [i2] (N, n_adcs, n_channels): pedestal value
            - ped_mae [f4] (N, n_adcs, n_channels): pedestal median-absolute error
            - diff_mae [f4] (N, n_adcs, n_channels): derivative median-absolute error (within pedestal window)
            - integral [f4] (N, n_adcs, n_channels): pedestal-subtracted full waveform integral
            - prompt_diff [f4] (N, n_adcs, n_channels): prompt derivative amplitude
            - prompt_t [f4] (N, n_adcs, n_channels): prompt rising-edge sample time (extrapolated)
            - prompt_sig [f4] (N, n_adcs, n_channels): prompt 20 sample gated integral
            - delayed_diff [f4] (N, n_adcs, n_channels): delayed derivative amplitude
            - delayed_t [f4] (N, n_adcs, n_channels): delayed rising-edge sample time
            - delayed_sig [f4] (N, n_adcs, n_channels): delayed 20 sample gated integral
            - delayed_ped [f4] (N, n_adcs, n_channels): pedestal used for delayed signal integration

Usage:

    ./light_event_builder.py --busy_channel=0 --n_adcs=2 --n_channels=64 --n_samples=256 \
        -i <input file> -o <output file> \
        -n <max events to process> \
        [--drop_wvfms] [--no_reco]

'''

# Process:
# Load all data from event (and next? - maybe just keep track of event loss at this point)
# Align ns based on busy_channel
# Create data:
#     event     - [u8] (N,)
#     sn        - [u8] (N, n_adcs)
#     ch        - [u1] (N, n_adcs, n_channels)
#     utime_ms  - [u8] (N, n_adcs)
#     tai_ns    - [y8] (N, n_adcs)
#     wvfm_valid - [u1] (N, n_adcs, n_channels) 1 if waveform present in event
#     wvfm      - [i2] (N, n_adcs, n_channels, n_samples)
#

import h5py
import numpy as np
import numpy.ma as ma
import ROOT
import root_numpy as rootnp
import os
from collections import defaultdict
import argparse
from argparse import RawDescriptionHelpFormatter
import time
from tqdm import tqdm

_default_busy_channel=0
_default_n_adcs=2
_default_n_channels=64
_default_n_samples=256
_default_n=None
_default_drop_wvfms=False
_default_no_reco=False
_default_skip=0

class BufferedEventFile(object):
    '''
        Class to handle periodic writing of data to the h5 file
    '''
    meta_name = 'light_meta'
    
    def __init__(self, filename, buffer_size=10240):
        '''
            :param filename: path and name of output file
            :param buffer_size: number of events to collect before writing
        '''
        if os.path.exists(filename):
            raise RuntimeError(f'{filename} already exists! Refusing to overwrite or append.')
        self.filename = filename
        self.buffer_size = buffer_size
        self._buffer = defaultdict(list)
        
    def write_meta(self, **meta):
        '''
            Update meta data (i.e. hdf5 attributes of "meta" group)
        '''
        with h5py.File(self.filename, 'a') as f:
            if self.meta_name not in f.keys():
                f.create_group(self.meta_name)
            for key,val in meta.items():
                f[self.meta_name].attrs[key] = val
  
    def fill(self, row, dset_name):
        '''
            Insert a row into the specified dataset
        '''
        self._buffer[dset_name].append(row)
        if len(self._buffer[dset_name]) > self.buffer_size:
            self.flush(dset_name)
            self._buffer[dset_name] = []
            
    def flush(self, dset_name=None):
        '''
            Flush any queued data to the hdf5 file
            
            :param dset_name: if not None, flush data only for this dataset
        '''
        if dset_name is None:
            dset_names = self._buffer.keys()
        else:
            dset_names = [dset_name]
        with h5py.File(self.filename, 'a') as f:
            self.write_meta(modified=time.time())
            for dset_name in dset_names:
                new_data = np.concatenate(self._buffer[dset_name], axis=0)
                if dset_name not in f.keys():
                    maxshape = list(new_data.shape)
                    maxshape[0] = None
                    f.create_dataset(dset_name, data=new_data,
                                     maxshape=maxshape,
                                     compression='gzip')
                else:
                    dset = f[dset_name]
                    newshape = list(dset.shape)
                    newshape[0] += new_data.shape[0]
                    dset.resize(newshape)
                    dset[newshape[0]-new_data.shape[0]:] = new_data
                
def intersection(y,x1,x2,y1,y2):
    '''
        Find the linear intersection between the line defined by (x1,y1), (x2,y2)
        and the horizontal line defined by (:,y)
    '''
    return (y * (x1-x2) + y1 * x2 - y2 * x1) / (y1 - y2)
                
def do_reco(base_array, wvfm_array, reco_array):
    '''
        Run a simple prompt/delayed reconstruction and fill the reco_array
    '''
    # run simplistic prompt/delayed reconstruction
    wvfm_shape = wvfm_array['samples'][0,:,:,:].shape
    
    # find primary rising edge
    diff = np.diff(wvfm_array['samples'].reshape(wvfm_shape), axis=-1) # Nserial, Nchan, Nsamples-1
    prompt_edge = (np.argmax(diff, axis=-1)).reshape(wvfm_shape[:-1] + (1,)) # Nserial, Nchan, 1
    
    # find pedestal (from data prior to rising edge)
    sample_idx = np.arange(wvfm_array['samples'].shape[-1]).reshape((1,1) + wvfm_shape[-1:]) # 1, 1, Nsamples
    prompt_mask = sample_idx > prompt_edge # Nserial, Nchan, Nsamples
    masked_wvfm = ma.array(wvfm_array['samples'].reshape(wvfm_shape), mask=prompt_mask) # Nserial, Nchan, Nsamples
    ped = ma.median(
        masked_wvfm,
        axis=-1, keepdims=True
    ) # Nserial, Nchan, 1
    ped_mae = ma.median(
        np.abs(masked_wvfm - ped),
        axis=-1, keepdims=True
    ) # Nserial, Nchan, 1
    diff_mae = ma.median(
        np.abs(ma.array(diff, mask=prompt_mask[:,:,1:])),
        axis=-1, keepdims=True
    ) # Nserial, Nchan, 1
    
    # find prompt time using linear projection
    y1 = np.take_along_axis(wvfm_array['samples'].reshape(wvfm_shape), prompt_edge, axis=-1) # Nserial, Nchan, 1
    y2 = np.take_along_axis(wvfm_array['samples'].reshape(wvfm_shape), prompt_edge+1, axis=-1) # Nserial, Nchan, 1
    prompt_time = intersection(ped, prompt_edge, prompt_edge+1, y1, y2) # Nserial, Nchan, 1
    
    # find delayed signal
    falling_edge = np.argmax(prompt_mask[:,:,1:] & (diff < 0), axis=-1).reshape(wvfm_shape[:-1] + (1,)) # Nserial, Nchan, 1
    delayed_mask = sample_idx[:,:,:-1] < falling_edge # Nserial, Nchan, Nsamples-1
    masked_diff = ma.array(diff, mask=delayed_mask) # Nserial, Nchan, Nsamples-1
    delayed_edge = (ma.argmax(masked_diff, axis=-1)).reshape(wvfm_shape[:-1] + (1,)) # Nserial, Nchan, 1
    delayed_mask = sample_idx > delayed_edge # Nserial, Nchan, Nsamples
    
    # find delayed time using linear projection - nix'd for the moment
    #y1 = np.take_along_axis(wvfm_array['samples'].reshape(wvfm_shape), delayed_edge, axis=-1) # Nserial, Nchan, 1
    #y2 = np.take_along_axis(wvfm_array['samples'].reshape(wvfm_shape), delayed_edge+1, axis=-1) # Nserial, Nchan, 1
    delayed_time = delayed_edge #intersection(ped, delayed_edge, delayed_edge+1, y1, y2)
    
    # calculate signal amplitude (gated integral)
    masked_wvfm.mask = ~(prompt_mask & (sample_idx < (prompt_time + 20))) # Nserial, Nchan, Nsamples
    prompt_sig = ma.sum(masked_wvfm - ped, axis=-1, keepdims=True) # Nserial, Nchan, 1
    # calculate delayed amplitude (also gated 20, but uses just prior sample as pedestal)
    masked_wvfm.mask = ~(delayed_mask & (sample_idx < delayed_edge + 20)) # Nserial, Nchan, Nsamples
    delayed_ped = np.take_along_axis(wvfm_array['samples'].reshape(wvfm_shape), delayed_edge-1, axis=-1) # Nserial, Nchan, 1
    delayed_sig = ma.sum(masked_wvfm - delayed_ped, axis=-1, keepdims=True) # Nserial, Nchan, 1
    
    reco_array['ped'][0] = ped[:,:,0]
    reco_array['ped_mae'][0] = ped_mae[:,:,0]
    reco_array['diff_mae'][0] = diff_mae[:,:,0]
    reco_array['integral'][0] = np.sum(wvfm_array['samples'].reshape(wvfm_shape) - ped, axis=-1)
    reco_array['prompt_diff'][0] = np.max(diff, axis=-1)
    reco_array['prompt_t'][0] = prompt_time[:,:,0]
    reco_array['prompt_sig'][0] = prompt_sig[:,:,0]
    reco_array['delayed_diff'][0] = ma.max(masked_diff, axis=-1)
    reco_array['delayed_t'][0] = delayed_time[:,:,0]
    reco_array['delayed_sig'][0] = delayed_sig[:,:,0]
    reco_array['delayed_ped'][0] = delayed_ped[:,:,0]
        
def compile_event(base_array, wvfm_array, reco_array, busy_channel=_default_busy_channel, no_reco=_default_no_reco):
    '''
        Process base array into an event (filling the wvfm, reco array)
    '''
    # estimate alignment for the arrays based on busy signal
    diff = np.diff(wvfm_array['samples'][0,:,busy_channel,:], axis=-1)
    base_array['alignment'][0,:] = np.argmax(diff, axis=-1) + 1
    
    if not no_reco:
        do_reco(base_array, wvfm_array, reco_array)
    
    return True

def store_entry(entry, event_buffer, dtype):
    '''
        Convert TTree entry into a numpy type - this is heckin slow....
    '''
    # create new row
    event = np.zeros((1,), dtype=dtype)
    
    event['event'] = entry.event
    event['sn'] = entry.sn
    event['ch'] = entry.ch
    event['utime_ms'] = entry.utime_ms
    event['tai_ns'] = entry.tai_ns
    event['wvfm'] = -rootnp.hist2array(entry.th1s_ptr)

    event_buffer[entry.sn].append(event)
    return event_buffer

def store_event(event_number, event_buffer, base_array, wvfm_array, reco_array):
    '''
        Pull from event buffers and assemble into event (fills base array and wvfm array)
        
        :returns: event_number, event_buffer : event_number will be incremented when full event has been assembled
    '''
    while all([len(buf) for buf in event_buffer.values()]):
        # check if data in buffers match (either the event or each other)
        sn = [key for key in event_buffer.keys() if len(event_buffer[key])]
        utime_ms = np.array([event_buffer[key][0][0]['utime_ms'] for key in sn]).astype(int)
        tai_ns = np.array([event_buffer[key][0][0]['tai_ns'] for key in sn]).astype(int)
        
#         print(list(zip(sn,utime_ms,tai_ns)))

        valid_mask = np.any(base_array['wvfm_valid'], axis=-1)
        if np.any(valid_mask):
#             print('data in event',np.argwhere(valid_mask))
            # existing data in event, check if new data matches
            event_ms = base_array['utime_ms'][valid_mask].astype(int)
            event_ns = base_array['tai_ns'][valid_mask].astype(int)

            match_idcs = np.argwhere(
                (np.abs(utime_ms-event_ms) <= 1000) & (np.abs(tai_ns-event_ns) <= 1000)
            ).flatten()
            if len(match_idcs):
#                 print('match to event', match_idcs)
                # there's a match (or more), so just grab one of them
                i = None
                for j in match_idcs:
                    event = event_buffer[sn[j]][0][0]
                    sn_mod = sn[j] % base_array['sn'].shape[-1]
                    ch_mod = event['ch'] % base_array['ch'].shape[-1]

                    if base_array['wvfm_valid'][0, sn_mod, ch_mod]:
                        continue
                    i = j
                if i is None:
#                     print('no room in event')
                    # no place for any data in current event, so declare a new event
                    return event_number + 1, event_buffer
            else:
#                 print('no match with event')
                # there's no match, so declare a new event
                return event_number + 1, event_buffer
        else:
            # no existing data in event, fill with earliest
            idcs = np.argsort(tai_ns)
#             print('no data in event',idcs)
            
            if len(idcs) > 1:
                # check for potential PPS rollover
                if np.any(np.diff(tai_ns[idcs].astype(int)) > 1e8) and np.any(np.abs(np.diff(utime_ms[idcs].astype(int))) < 500):
                    i = idcs[-1]
                # check for significant time offset
                if np.any(np.abs(np.diff(utime_ms[idcs].astype(int))) > 500):
                    i = np.argsort(utime_ms)[0]
                # default to tai ns ordering
                else:
                    i = idcs[0]
            else:
                i = idcs[0]
#         print('fill',sn[i])

        event = event_buffer[sn[i]][0][0]
        sn_mod = sn[i] % base_array['sn'].shape[-1]
        ch_mod = event['ch'] % base_array['ch'].shape[-1]

        # fill base array
        base_array['event'] = event_number
        base_array['sn'][0, sn_mod] = event['sn']
        base_array['ch'][0, sn_mod, ch_mod] = event['ch']
        base_array['utime_ms'][0, sn_mod] = event['utime_ms']
        base_array['tai_ns'][0, sn_mod] = event['tai_ns']
        base_array['wvfm_valid'][0, sn_mod, ch_mod] = True

        # fill waveform array
        wvfm_array['samples'][0, sn_mod, ch_mod] = event['wvfm']

        # remove from buffer
        event_buffer[sn[i]] = event_buffer[sn[i]][1:]
                
    return event_number, event_buffer


        
def main(input_file, output_file, **kwargs):
    n = kwargs.pop('n', _default_n)
    
    # create new event file
    buffered_file = BufferedEventFile(output_file)
    buffered_file.write_meta(created=time.time())
    buffered_file.write_meta(**kwargs, source_file=input_file, n=n if n is not None else -1)
    
    # create waveform event buffer
    n_samples = kwargs.get('n_samples', _default_n_samples)
    event_buffer_dtype = np.dtype([
        ('event', 'i4'),
        ('sn', 'i4'),
        ('ch', 'i4'),
        ('utime_ms', 'i8'),
        ('tai_ns', 'i8'),
        ('wvfm', 'i2', n_samples)
    ])
    
    # create event datatype based on arguments
    n_adcs = kwargs.get('n_adcs', _default_n_adcs)
    n_channels = kwargs.get('n_channels', _default_n_channels)
    base_dtype = np.dtype([
        ('event', 'u8'),
        ('sn', 'u8', n_adcs),
        ('ch', 'u1', (n_adcs, n_channels)),
        ('utime_ms', 'u8', n_adcs),
        ('tai_ns', 'u8', n_adcs),
        ('alignment', 'i4', n_adcs),
        ('wvfm_valid', 'u1', (n_adcs, n_channels))
    ])
    
    # put waveforms in a separate dataset for better access / memory usage
    drop_wvfms = kwargs.get('drop_wvfms', _default_drop_wvfms)
    wvfm_dtype = np.dtype([
        ('samples', 'i2', (n_adcs, n_channels, n_samples))
    ])
    
    # stuff I want for reconstructing the michel time spectrum
    no_reco = kwargs.get('no_reco', _default_no_reco)
    reco_dtype = np.dtype([
        ('ped', 'i2', (n_adcs, n_channels)),
        ('ped_mae', 'f4', (n_adcs, n_channels)),
        ('diff_mae', 'f4', (n_adcs, n_channels)),
        ('integral', 'f4', (n_adcs, n_channels)),
        ('prompt_diff', 'f4', (n_adcs, n_channels)),
        ('prompt_t', 'f4', (n_adcs, n_channels)),
        ('prompt_sig', 'f4', (n_adcs, n_channels)),
        ('delayed_diff', 'f4', (n_adcs, n_channels)),
        ('delayed_t', 'f4', (n_adcs, n_channels)),
        ('delayed_sig', 'f4', (n_adcs, n_channels)),
        ('delayed_ped', 'f4', (n_adcs, n_channels)),
    ])
    
    event_buffer = defaultdict(list)
    base_array = np.zeros((1,), dtype=base_dtype)
    wvfm_array = np.zeros((1,), dtype=wvfm_dtype)
    reco_array = np.zeros((1,), dtype=reco_dtype)
    
    # prep loop
    root_file = ROOT.TFile(input_file)
    rwf = root_file.Get('rwf')
    entries = rwf.GetEntries()
    entry = 0
    curr_event = 0
    total_events = 0
    busy_channel = kwargs.get('busy_channel', _default_busy_channel)
    skip = kwargs.get('skip', _default_skip)
    entry += skip
    subloop_flag = True
    with tqdm(total=n if n is not None else entries-entry) as pbar:
        while (entry < entries) or all([len(buf) for buf in event_buffer.values()]):            
            while entry < entries and subloop_flag:
                # get new data while it's present
                rwf.GetEntry(entry)
                if n is None: pbar.update(1)
                entry += 1

                # add entry to event buffer
                event_buffer = store_entry(rwf, event_buffer, event_buffer_dtype)
                
                # get data until data present in all buffers
                subloop_flag = not all([len(buf) for buf in event_buffer.values()])
            pbar.set_description(str([(sn,len(buf),buf[0]['utime_ms'] if len(buf) else '-') for sn,buf in event_buffer.items()]), refresh=False)
                
            # combine into events
            new_event, event_buffer = store_event(curr_event, event_buffer, base_array, wvfm_array, reco_array)

            # keep track of the event
            if new_event != curr_event:
                total_events += 1

                # compile event (alignment, prompt/delayed, yadda yadda...)
                if compile_event(base_array, wvfm_array, reco_array, busy_channel=busy_channel, no_reco=no_reco):

                    # new event, store old arrays
                    buffered_file.fill(base_array, 'light_event')
                    if not no_reco:
                        buffered_file.fill(reco_array, 'light_reco')
                    if not drop_wvfms:
                        buffered_file.fill(wvfm_array, 'light_wvfm')

                if n is not None:
                    pbar.update(1)
                    if total_events >= n:
                        break

                # reset arrays
                base_array = np.zeros((1,), dtype=base_dtype)
                wvfm_array = np.zeros((1,), dtype=wvfm_dtype)
                reco_array = np.zeros((1,), dtype=reco_dtype)
                
            # update position
            curr_event = new_event
            subloop_flag = True
        
        buffered_file.flush()
    print('Done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--busy_channel', type=int, default=_default_busy_channel, help='''busy channel for alignment (default=%(default)s)''')
    parser.add_argument('--n_adcs', type=int, default=_default_n_adcs, help='''number of adcs (default=%(default)s)''')
    parser.add_argument('--n_channels', type=int, default=_default_n_channels, help='''number of channels per adc (default=%(default)s)''')
    parser.add_argument('--n_samples', type=int, default=_default_n_samples, help='''number of samples per waveform (default=%(default)s)''')
    parser.add_argument('--input_file','-i', type=str, required=True)
    parser.add_argument('--output_file','-o', type=str, required=True)
    parser.add_argument('--drop_wvfms', action='store_true', help='''if present, do not store waveforms in output file''')
    parser.add_argument('--no_reco', action='store_true', help='''if present, do not run prompt/delayed reconstruction in output file''')
    parser.add_argument('--n','-n', type=int, default=_default_n, help='''max events to process (default=%(default)s)''')
    parser.add_argument('--skip', type=int, default=_default_skip, help='''skip this many entries to start - note: if used, first event in file may be incomplete (default=%(default)s)''')
    args = parser.parse_args()
    main(**vars(args))
