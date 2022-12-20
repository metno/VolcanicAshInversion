#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import logging
import pprint
import time
from netCDF4 import Dataset, num2date
import os
import re
import json
import tempfile

from cartopy import crs as ccrs
from AshInv import Misc, Plot

import time


if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgParser(description='Generate synthetic satellite image from simulations.')
    parser.add("--verbose", action="store_true", help="Enable verbose mode")
    
    parser.add('--a_posteriori_filename', type=str,
                        help='JSON-file from an AshInv simulation', required=True)
    parser.add('--a_posteriori_varname', type=str,
                        help='Variable in JSON file to generate satellite image from (typically a_priori_2d or a_posteriori_2d)', required=False, default='a_posteriori_2d')
    parser.add('--output_filename', type=str,
                        help='Output filename to save (numpy npz-file)', required=True)
    parser.add('--emep_runs', type=str,
                        help="CSV-file with EEMEP simulation runs", required=True)
    parser.add('--emep_runs_basedir', type=str,
                        help='Absolute path to where eemep simulation files are', required=True)

    args = parser.parse_args()

    print("Arguments: ")
    for var, val in vars(args).items():
        print("{:s} = {:s}".format(var, str(val)))

    #Helper class to give some progress info
    class Timer(object):
        def __init__(self, tag):
            self.tag = tag

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.secs = self.end - self.start
            self.msecs = self.secs * 1000 # millisecs
            print("{:s}: {:f} ms".format(self.tag, self.msecs))

    #Read input json
    json_data = Plot.readJson(args.a_posteriori_filename)
    #print(json_data.keys())

    #Read input eemep files
    sim_files = pd.read_csv(args.emep_runs, parse_dates=[0], comment='#')
    sim_files.filename = sim_files.filename.apply(lambda x: os.path.join(args.emep_runs_basedir, x))
    sim_files['date'] = sim_files['date'].dt.tz_convert(None)
    sim_files_missing = sim_files['filename'].apply(os.path.exists) == False
    sim_files.drop(sim_files[sim_files_missing].index, inplace=True)
    sim_files = sim_files.reset_index(drop=True)
    print(sim_files)

    # Find the index of into sim_files for each emission_time so that we can match 
    # each emission time with the right sim file.
    sim_times = np.array(sim_files.date, dtype='datetime64[ns]')
    timesteps = np.zeros((len(json_data['emission_times'])), dtype=np.int64)
    for timestep in range(len(timesteps)):
        t = json_data['emission_times'][timestep]
        t_index = np.argmin(np.abs(t - sim_times))

        #Find file matching emission time
        if ((t - sim_times[t_index]) / np.timedelta64(1, 'm') > 5):
            print("Warning: Could not find eemep simulation file matching time {:s}".format(t))
            timesteps[timestep] = -1
            continue
        else:
            timesteps[timestep] = t_index
    print(timesteps)

    #Read the right variable to generate satellite image from
    a_priori_2d = np.array(json_data[args.a_posteriori_varname], dtype=np.float64)
    ordering_index = np.array(json_data["ordering_index"])
    print(ordering_index.shape)
    print(ordering_index[:,0])
    a_priori_2d = a_priori_2d[ordering_index]
    print(a_priori_2d[:,0])

    #Generate output times
    start_time = json_data['emission_times'].min()
    end_time = json_data['emission_times'].max() + np.timedelta64(6,'D')
    hours = (end_time-start_time) / np.timedelta64(1,'h')
    output_times = np.arange(start_time, end_time, np.timedelta64(1,'h'))
    print(len(output_times))

    #Set output filename to store data in
    print("Using {:s}".format(args.output_filename))

    if os.path.exists(args.output_filename):
        print("Continuing existing run by reading data")
        existing_data = np.load(args.output_filename, allow_pickle=True)
        output_data = existing_data['output_data']
        a_priori_2d = existing_data['a_priori_2d']
        varname = existing_data['varname']
        json_filename = existing_data['json_filename']
        skip_iterations = existing_data['i']
        args_str = existing_data['args']
        assert(args_str == str(args))
        print("varname: " + varname)
        print("json_filename: " + json_filename)
    else:
        print("Creating new file")
        output_data = None
        skip_iterations = 0

    #Helper function that safely saves to file
    def save_to_file(output_filename, **kwargs):
        temp = tempfile.NamedTemporaryFile(delete=False)
        bakfile = output_filename + ".bak"
        np.savez_compressed(temp, **kwargs)
        temp.close
        if (os.path.exists(bakfile)):
            os.remove(bakfile)
        if (os.path.exists(output_filename)):
            os.rename(output_filename, bakfile)
        os.rename(temp.name, output_filename)

    #Loop over each eemep file and vertically integrate by multiplying with a priori/a posteriori
    tic = time.time()
    for i, timestep in enumerate(timesteps):
        if (i < skip_iterations):
            print("Skipping {:d}: already processed".format(i))
            continue

        if timestep>=0:
            filename = sim_files['filename'][timestep]

            timertag = "{:d} ({:d}/{:d}) - {:s}".format(timestep, i, len(timesteps), filename)
            with Timer(timertag) as t:
                with Dataset(filename) as nc_file:
                    #Get timesteps within this nc_file
                    sim_time = nc_file['time'][:]
                    unit = nc_file['time'].units
                    sim_time = num2date(sim_time, units = unit, only_use_cftime_datetimes=False, only_use_python_datetimes=True).astype('datetime64[ns]')

                    if (output_times is None):
                        output_times = sim_time
                    if (output_data is None):
                        output_data_shape = list(nc_file['COLUMN_ASH_L01_k20'].shape)
                        output_data_shape[0] = len(output_times)
                        output_data = np.zeros(output_data_shape)

                    #Find which indices these correspond to in our output timesteps
                    output_indices = np.where(np.in1d(output_times, sim_time))[0]
                    input_indices = np.where(np.in1d(sim_time, output_times))[0]

                    #Skip files that do not contribute to output
                    if ((len(input_indices) > 0) and (len(output_indices) > 0)):
                        #Initialize sim_data to zeros
                        sim_data_shape = list(nc_file['COLUMN_ASH_L01_k20'].shape)
                        sim_data_shape[0] = len(input_indices)
                        sim_data = np.zeros(sim_data_shape)

                        for level in range(1,20):
                            a_priori_value = a_priori_2d[20-level-1, timestep-1]
                            #Skip reading data that is not going to be in the a priori
                            if (np.abs(a_priori_value) > 0):
                                varname = "COLUMN_ASH_L{:02d}_k20".format(level)
                                #Levels start with 0 at the top, whilst a-priori starts with 0 at ground level.
                                sim_data += nc_file[varname][input_indices,:,:] * a_priori_value

                        output_data[output_indices,:,:] += sim_data
                    else:
                        print("No relevant data, skipping")

        #Save to file in a "safe" manner every 5 minutes
        if (time.time() - tic > 300):
            print("Saving to file")
            save_to_file(args.output_filename, 
                         output_data=output_data, 
                         i=i, 
                         a_priori_2d=a_priori_2d,
                         varname=args.a_posteriori_varname,
                         json_filename=args.a_posteriori_filename,
                         args=str(args))
            tic=time.time()

    save_to_file(args.output_filename, 
                 output_data=output_data, 
                 i=i, 
                 a_priori_2d=a_priori_2d,
                 varname=args.a_posteriori_varname,
                 json_filename=args.a_posteriori_filename,
                 args=str(args))

    print("Done")
