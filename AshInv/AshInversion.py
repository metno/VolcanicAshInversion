#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#    This file is part of PVAI - Python Volcanic Ash Inversion.              #
#                                                                            #
#    Copyright 2019, 2020 The Norwegian Meteorological Institute             #
#               Authors: André R. Brodtkorb <andreb@met.no>                  #
#                                                                            #
#    PVAI is free software: you can redistribute it and/or modify            #
#    it under the terms of the GNU General Public License as published by    #
#    the Free Software Foundation, either version 2 of the License, or       #
#    (at your option) any later version.                                     #
#                                                                            #
#    PVAI is distributed in the hope that it will be useful,                 #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#    GNU General Public License for more details.                            #
#                                                                            #
#    You should have received a copy of the GNU General Public License       #
#    along with PVAI. If not, see <https://www.gnu.org/licenses/>.           #
#                                                                            #
##############################################################################

import numpy as np
import pandas as pd
import datetime
import logging
import pprint
import time
import netCDF4
import glob
import os
import re
import json
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.optimize
import gc
import psutil
import importlib

#Enable this to profile memory
#from memory_profiler import profile


from matplotlib import pyplot as plt
import matplotlib.colors

from AshInv import Plot, Misc

class AshInversion():
    def __init__(self, verbose=0):
        """
        Constructor

        verbose - adjust verbosity of program [0-100]
        """
        self.logger = logging.getLogger(__name__)

        self.args = {}
        self.args['verbose'] = verbose




    #@profile
    def make_ordering_index(self, mask=None):
        """
        Create linear ordering of matrix elements.
        This maps an emission altitude and emission time to a linear index
        """
        #Generate matrix ordering index
        no_data_value = -9999
        self.logger.debug("Creating matrix order")
        num_altitudes = self.level_heights.size
        num_timesteps = self.emission_times.size
        if not hasattr(self, 'ordering_index'):
            self.ordering_index = np.zeros((num_altitudes, num_timesteps), dtype=np.int64)

        idx = 0
        delete_idx = 0
        to_delete = []
        for t in range(num_timesteps):
            for a in range(num_altitudes):
                #Previously masked element
                if (self.ordering_index[a, t] == no_data_value):
                    pass
                #Unmasked element - keep
                elif (mask is None or mask[a, t] == False):
                    self.ordering_index[a, t] = idx
                    idx += 1
                    delete_idx += 1
                #newly masked element - mask and remove from system
                else:
                    self.ordering_index[a, t] = no_data_value
                    to_delete += [delete_idx]
                    delete_idx += 1

        if hasattr(self, 'M'):
            to_keep = np.ones((self.M.shape[1]), dtype=np.bool)
            to_keep[to_delete] = False
            to_keep = np.flatnonzero(to_keep)
            self.M = self.M[:,to_keep]
        if hasattr(self, 'Q'):
            to_keep = np.ones((self.Q.shape[1]), dtype=np.bool)
            to_keep[to_delete] = False
            to_keep = np.flatnonzero(to_keep)
            self.Q = self.Q[:,to_keep]
            self.Q = self.Q[to_keep,:]
        if hasattr(self, 'B'):
            to_keep = np.ones((self.Q.shape[1]), dtype=np.bool)
            to_keep[to_delete] = False
            to_keep = np.flatnonzero(to_keep)
            self.B = self.B[to_keep]

        self.x_a = np.delete(self.x_a, to_delete)
        self.sigma_x = np.delete(self.sigma_x, to_delete)





    #@profile
    def make_a_priori(self,
                        filename,
                        scale_a_priori=1.0e0,
                        min_emission_uncertainty_scale=5.0e-2,
                        min_emission_scale=2.0e-2,
                        min_emission=2.0e-5,
                        min_emission_error=1.0e-5):
        """
        Create left hand side vector of emissions and emission uncertainties

        Read the a priori emission estimates from file
        and make the a priori vectors

        param: a_priori_file - CSV-file with dates and filenames for satellite observations

        scale_a_priori - Scale a_priori to same unit as emission/observation (typically kg => kg is 1)
        min_emission_uncertainty_scale - Set minimum uncertainty to average multiplied by scale
        min_emission_scale - Set minimum emission to average multiplied by scale", default=2e-2)
        """
        #Read from file
        with open(filename, 'r') as infile:
            #Read file
            tmp = json.load(infile)

            #Copy only data we care about
            self.emission_times = np.array(tmp["emission_times"], dtype='datetime64[ns]')
            self.level_heights = np.array(tmp["level_heights"], dtype=np.float64)
            self.volcano_altitude = tmp["volcano_altitude"]
            a_priori_2d = np.array(tmp["a_priori_2d"], dtype=np.float64)
            a_priori_2d_uncertainty = np.array(tmp["a_priori_2d_uncertainty"], dtype=np.float64)

            #a_priori_2d[:,:] = a_priori_2d[:,:]*2.0
            #a_priori_2d = a_priori_2d * (1.0 + 0.25 * (np.random.random(a_priori_2d.shape) - 0.5))
            #a_priori_2d[:8,:] *= 0.25
            #a_priori_2d[8:,:] *= 2.0

            #Free file data
            tmp = None

        self.logger.debug("{:d} altitudes and {:d} timesteps, {:d} a priori values".format(self.level_heights.size, self.emission_times.size, a_priori_2d.size))

        num_timesteps = self.emission_times.size
        num_altitudes = self.level_heights.size

        #Create vectors if empty
        if not hasattr(self, 'x_a'):
            self.logger.info("Creating new vectors and ordering index")
            self.x_a = np.empty((num_altitudes*num_timesteps), dtype=np.float64)
            self.sigma_x = np.empty((num_altitudes*num_timesteps), dtype=np.float64)
            self.make_ordering_index()

        #Create the left hand side vectors
        for a in range(num_altitudes):
            for t in range(num_timesteps):
                emission_index = self.ordering_index[a, t]

                if (emission_index >= 0):
                    self.x_a[emission_index] = a_priori_2d[t, a]
                    self.sigma_x[emission_index] = a_priori_2d_uncertainty[t, a]

        #Scale a priori and a priori standard deviation to correct unit
        x_a_min = np.maximum(self.x_a.mean(), min_emission)*min_emission_scale
        sigma_x_min = np.maximum(self.sigma_x.mean(), min_emission_error)*min_emission_uncertainty_scale
        self.x_a = np.maximum(self.x_a, x_a_min) * scale_a_priori
        self.sigma_x = np.maximum(self.sigma_x, sigma_x_min) * scale_a_priori


    #@profile
    def get_memory_usage_gb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024*1024*1024)


    #@profile
    def make_system_matrix(self,
                           matched_file_csv_filename,
                           scale_emission=1.0e-9,
                           scale_observation=1.0e-3,
                           obs_zero=1.0e-20,
                           obs_model_error=1.0e-2,
                           obs_min_error=5.0e-3,
                           obs_zero_error_scale=1e2,
                           use_elevations=False,
                           store_full_matrix=False,
                           progress_file=None):
        """
        Read the matched data and create the system matrix

        scale_emission - Scale emission to same unit as a priori/observation (typically ug => kg is 1e-9)
        scale_observation - Scale observation to same unit as a priori/emission (typically g => kg is 1e-3)
        obs_zero - Observations smaller than this are treated as zero (with higher error)
        obs_model_error - Error (sigma) of observations
        obs_min_error - Minimum error (sigma) of observations
        obs_zero_error_scale - Scaling of error for zero observations

        matched_file_csv_filename - filename of matched files (generated by matchfiles)
        use_elevations - Use elevation information (if available) in the inversion
        """

        matched_file_dir = os.path.dirname(matched_file_csv_filename)
        matched_files_df = pd.read_csv(matched_file_csv_filename, parse_dates=[0], comment='#')
        if (self.args['verbose'] > 90):
            self.logger.debug("Matched files: " + pprint.pformat(matched_files_df))

        #Compute total number of emissions for each matched file
        matched_files_df["num_emissions"] = matched_files_df["num_observations"] \
                                            * matched_files_df["num_timesteps"] \
                                            * matched_files_df["num_altitudes"]
        total_num_obs = matched_files_df["num_observations"].sum()
        total_num_emis = matched_files_df["num_emissions"].sum()
        num_files = matched_files_df.shape[0]
        self.logger.info("Processing {:d} observations and {:d} emissions in {:d} files".format(total_num_obs, total_num_emis, num_files))

        #Check if we have observed altitudes
        #Then we double the number of observations: ash up to altitude, no ash above
        level_altitudes = np.cumsum(self.level_heights) + self.volcano_altitude
        try:
            nc_filename = os.path.join(matched_file_dir, matched_files_df.matched_file[0])
            nc_file = netCDF4.Dataset(nc_filename)
            if (use_elevations and 'obs_alt' in nc_file.variables.keys()):
                self.logger.info("Using altitudes in inversion")
                total_num_obs *= 2
            else:
                use_elevations = False
                self.logger.info("Altitudes not used for inversion")
        except Exception as e:
            self.logger.error("Could not open NetCDF dataset {:s}".format(nc_filename))
            raise e
        finally:
            nc_file.close()
            nc_file = None

        def npTimeToDatetime(np_time):
            return datetime.datetime.utcfromtimestamp((np_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

        #Remove timesteps from a priori that don't match up with matched files (make system matrix far smaller)
        matched_times = np.array(matched_files_df["date"], dtype='datetime64[ns]')
        remove = np.ones((self.level_heights.size, self.emission_times.size), dtype=np.bool)
        for t, emission_time in enumerate(self.emission_times):
            #FIXME: These should correspond to the limits in MatchFiles.py
            min_days=0
            max_days=6
            deltas = (matched_times - emission_time) / np.timedelta64(1, 'D')
            valid = (deltas >= min_days) & (deltas < max_days)
            if np.any(valid):
                remove[:,t] = False
        self.logger.info("Removing {:d}/{:d} a priori values without observations".format(np.count_nonzero(remove), self.emission_times.size*self.level_heights.size))
        self.make_ordering_index(remove)

        #Allocate data
        gc.collect()
        num_emissions = np.count_nonzero(self.ordering_index >= 0)
        timesteps_per_day = 24 / (np.diff(self.emission_times).max()/np.timedelta64(1, 'h'))
        assert num_emissions == (self.ordering_index.max()+1), "Ordering index has wrong entries!"
        self.logger.info("Memory used pre alloc: {:.1f} GB".format(self.get_memory_usage_gb()))

        #This forces allocation and initialization of data
        #Really important for speed. Without this, numpy appears to
        #reallocate and move all the data around
        nnz_counter = 0
        if (store_full_matrix):
            self.logger.info("Storing matrix M explicitly")
            self.logger.info("Will try to allocate {:d}x{:d} matrix with {:d} non-zeros (~{:.1f} GB)".format(
                    total_num_obs,
                    num_emissions,
                    total_num_emis,
                    total_num_emis*(8+8)/(1024*1024*1024)))
            #Allocated this way on purpose to force numpy to preallocate data for us
            M_data = np.empty(total_num_emis, dtype=np.float64)
            M_indices = np.empty(total_num_emis, dtype=np.int64)
            M_indptr = np.empty(total_num_obs+1, dtype=np.int64)
            M_data.fill(0.0)
            M_indices.fill(0)
            M_indptr.fill(0)

        #LSQR system matrix Q and right hand side B
        self.Q = np.zeros((num_emissions, num_emissions), dtype=np.float64)
        self.B = np.zeros((num_emissions), dtype = np.float64)
        Q_c = np.zeros_like(self.Q)
        B_c = np.zeros_like(self.B)
        sum_counter = 0

        #Right hand sides
        self.y_0 = np.zeros((total_num_obs), dtype=np.float64)
        self.sigma_o = np.zeros((total_num_obs), dtype=np.float64)
        self.logger.info("Memory used post alloc: {:.1f} GB".format(self.get_memory_usage_gb()))

        #Loop over all observations and assemble row by row into matrix
        n_removed = 0
        obs_counter = 0
        current_index = 0

        #Read existing progress file
        if (progress_file is not None and os.path.exists(progress_file)):
            extra_data = self.load_matrices(progress_file)
            n_removed = extra_data['n_removed']
            obs_counter = extra_data['obs_counter']
            current_index = extra_data['current_index']
            nnz_counter = extra_data['nnz_counter']
            Q_c = extra_data['Q_c']
            B_c = extra_data['B_c']

        start_assembly = time.time()
        next_save_time = time.time()
        run_eta = None
        for row_index, row in enumerate(matched_files_df.itertuples()):
            #Skip already processed rows
            if (row_index < current_index):
                self.logger.info("Skipping row {:d}".format(row_index))
                continue

            filename = row.matched_file
            filename = os.path.join(matched_file_dir, filename)
            if (self.args['verbose'] > 50):
                self.logger.debug("Reading {:s}".format(row.matched_file))

            if(row.num_observations == 0):
                continue

            timers = { 'start': {}, 'end': {} }
            timers['start']['tot'] = time.time()

            try:
                timers['start']['dsk'] = time.time()

                #NetCDF4 leaks stuff. Try reloading every n-th file
                if (row_index % 10 == 0):
                    importlib.reload(netCDF4)
                    gc.collect()

                nc_file = netCDF4.Dataset(filename)
                #Get the indices for the time which corresponds to the a-priori data
                unit = nc_file['time'].units
                times = netCDF4.num2date(nc_file['time'][:], units = unit).astype('datetime64[ns]')
                time_index = [[], []] #Two dimensional array [sim_time_index, emission_time_index]
                for i, t in enumerate(times):
                    ix = np.flatnonzero(self.emission_times == t)
                    if (len(ix) == 0):
                        self.logger.debug("Time {:s} from {:s} does not match up with a priori emission!".format(str(t), filename))
                    elif (len(ix) == 1):
                        time_index[0] += [i]
                        time_index[1] += [ix]
                    else:
                        self.logger.warning("Time {:s} exists multiple times in a priori - using first!".format(str(t)))
                        time_index[0] += [i]
                        time_index[1] += [ix[0]]

                if (self.args['verbose'] > 70):
                    self.logger.debug("File times: {:s}, \na priori times: {:s}, \nindices: {:s}".format(str(times), str(self.emission_times), str(time_index)))
                if (self.args['verbose'] > 90):
                    self.logger.debug("Time indices: {:s}".format(str(time_index)))

                #Get observations that are unmasked
                obs = nc_file['obs'][:]
                n_obs = len(obs)

                obs_flag = nc_file['obs_flag'][:]

                if (use_elevations):
                    obs_alt = nc_file['obs_alt'][:]*1000 #FIXME: Given in KM

                #Read simulation data from disk
                sim = nc_file['sim'][:,:,:]
                timers['end']['dsk'] = time.time()
            except Exception as e:
                self.logger.error("Could not open NetCDF file {:s}".format(filename))
                raise e
            finally:
                nc_file.close()
                nc_file = None

            #Check sizes
            assert (n_obs == row.num_observations), "Number of unmasked observations {:d} does not match up with expected {:d}".format(n_obs, row.num_observations)
            assert (row.num_observations == sim.shape[0]), "Number of observations {:d} does not match expected {:d}".format(row.num_observations, sim.shape[0])
            assert (row.num_timesteps == sim.shape[1]), "Number of timesteps {:d} does not match up to number of sims {:d}".format(row.num_timesteps, sim.shape[1])
            assert (row.num_altitudes == sim.shape[2]), "Number of altitudes {:d} does not match up to number of sims {:d}".format(row.num_altitudes, sim.shape[2])
            assert (row.num_altitudes == self.level_heights.size), "Number of altitudes does not match a priori!"
            if (self.args['verbose'] > 50):
                self.logger.debug("#obs={:d} ({:d}/{:d}), #time={:d}, #alt={:d}".format(n_obs, obs_counter, total_num_obs, row.num_timesteps, row.num_altitudes))

            #Initialize timers for assembly loop
            timers['start']['asm0'] = 0
            timers['end']['asm0'] = 0
            timers['start']['asm1'] = 0
            timers['end']['asm1'] = 0
            timers['start']['asm2'] = 0
            timers['end']['asm2'] = 0
            if (store_full_matrix):
                timers['start']['asm3'] = 0
                timers['end']['asm3'] = 0

            #  Assemble system matrix
            timers['start']['asm'] = time.time()
            max_i_width = 0
            for o in range(n_obs):
                #Set observation and uncertainty of observation
                self.y_0[obs_counter] = obs[o]*scale_observation
                current_obs_min_error = obs_min_error if (self.y_0[obs_counter] > obs_zero) else obs_min_error*obs_zero_error_scale
                current_sigma_o = obs[o]*scale_observation*1.0 #FIXME: constant uncertainty
                self.sigma_o[obs_counter] = np.sqrt(current_sigma_o**2 + obs_model_error**2 + current_obs_min_error**2)

                #Add observation of ash cloud top (zero ash above top of cloud)
                altitude_ranges = [slice(0, row.num_altitudes)]
                if (use_elevations):
                    #Make a duplicate observation at same lat lon, but with zero above the given altitude
                    self.y_0[obs_counter+1] = 0
                    #FIXME: Use small sigma here => assume height information is reliable
                    #Should perhaps be updated to be a function from cloud top observatoin?
                    self.sigma_o[obs_counter+1] = np.sqrt(obs_model_error**2 + obs_min_error**2)
                    altitude_max = min(row.num_altitudes, np.searchsorted(level_altitudes, obs_alt[o]))
                    altitude_ranges = [slice(0, altitude_max), slice(altitude_max, row.num_altitudes)]

                for altitude_range in altitude_ranges:
                    #Find valid values and indices
                    #Valid = not masked index && value > 0 && not masked
                    timers['start']['asm0'] += time.time()
                    vals = sim[o, time_index[0], altitude_range].ravel(order='C')
                    indices = self.ordering_index[altitude_range, time_index[1]].transpose().ravel(order='C')

                    assert (vals.shape == indices.shape), "Number of values {:d} does not match up with number of indices {:d}".format(str(vals.shape), str(indices.shape))

                    valid_vals = np.flatnonzero(indices >= 0)
                    valid_vals = valid_vals[vals[valid_vals] > 0.0]

                    vals = vals[valid_vals]*scale_emission
                    indices = indices[valid_vals]
                    timers['end']['asm0'] += time.time()

                    if (indices.size > 0):
                        #Compute matrix product Q = M^T sigma_o^-2 M without storing M
                        #Uses clever (basic) indexing to avoid superfluous copying of data
                        timers['start']['asm1'] += time.time()
                        i_min = indices.min()
                        i_max = indices.max()+1
                        max_i_width = max(i_max-i_min-1, max_i_width)
                        v = np.zeros(i_max - i_min)
                        v[indices-i_min] = vals
                        V = np.outer(v/(self.sigma_o[obs_counter]**2), v)
                        timers['end']['asm1'] += time.time()
                        timers['start']['asm2'] += time.time()
                        Q_c[i_min:i_max, i_min:i_max] += V
                        timers['end']['asm2'] += time.time()

                        #Compute matrix product B = M^t sigma_o^-2 (y_0 - M x_a) without storing M
                        B_c[i_min:i_max] += v*(self.y_0[obs_counter] - np.dot(v, self.x_a[i_min:i_max])) / (self.sigma_o[obs_counter]**2)

                        if (store_full_matrix):
                            timers['start']['asm3'] += time.time()
                            nnz_next = nnz_counter+vals.size
                            M_indices[nnz_counter:nnz_next] = indices
                            M_data[nnz_counter:nnz_next] = vals
                            M_indptr[obs_counter+1] = vals.size
                            nnz_counter = nnz_next
                            timers['end']['asm3'] += time.time()

                    obs_counter = obs_counter+1


            #Increase accuracy by grouping additions
            #(ref Kahan summation and rounding)
            sum_counter += n_obs
            last_row = row_index+1 == matched_files_df.shape[0]
            if ((sum_counter > 1.0e5) or last_row):
                if  not last_row:
                    valid_Q = (self.Q != 0.0)
                    if (np.sum(valid_Q) > 0):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            # self.Q might be zero initially
                            valid_Q = np.logical_and(Q_c != 0.0, valid_Q, np.nan_to_num(Q_c / self.Q) > 1.0e-5)
                    else:
                        valid_Q = Q_c > 0.0

                    valid_B = self.B != 0.0
                    if (np.sum(valid_B) > 0):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            # self.B might be zero initially
                            valid_B = np.logical_and(B_c != 0.0, valid_B, np.nan_to_num(B_c / self.B) > 1.0e-5)
                    else:
                        valid_B = (B_c != 0.0)
                else:
                    valid_Q = (Q_c != 0.0)
                    valid_B = (B_c != 0.0)

                num_valid_Q = np.sum(valid_Q)
                num_valid_B = np.sum(valid_B)

                self.logger.info("#valid Q={:d}".format(num_valid_Q))
                self.logger.info("#valid B={:d}".format(num_valid_B))

                if (num_valid_Q > 0):
                    self.Q[valid_Q] += Q_c[valid_Q]
                    Q_c[valid_Q] = 0.0

                if (num_valid_B > 0):
                    self.B[valid_B] += B_c[valid_B]
                    B_c[valid_B] = 0.0

                sum_counter = 0

            timers['end']['asm'] = time.time()
            timers['end']['tot'] = time.time()

            #Save progress every 5 minutes (for restarting jobs)
            if ((progress_file is not None) and (time.time() > next_save_time)):
                self.logger.info("Writing progress file for restarting")
                current_index = row_index+1
                extra_data =  {
                    'Q_c': Q_c,
                    'B_c': B_c,
                    'obs_counter': obs_counter,
                    'n_removed': n_removed,
                    'current_index': current_index,
                    'nnz_counter': nnz_counter
                }
                self.save_matrices(progress_file, extra_data)
                next_save_time = time.time() + 5*60

            logstr = []
            new_run_eta = ((timers['end']['tot'] - timers['start']['tot'])/n_obs)*(total_num_obs-obs_counter)
            if (run_eta is None):
                run_eta = new_run_eta
            else:
                run_eta = 0.9*run_eta + 0.1*new_run_eta
            logstr += ["i={:d}".format(row_index)]
            logstr += ["ETA {:s}".format(str(datetime.timedelta(seconds=run_eta)))]
            logstr += ["{:.1f} %".format(100*obs_counter/total_num_obs)]
            logstr += ["#obs={:d}".format(n_obs)]
            logstr += ["#obs/s={:.1f}".format(n_obs/(timers['end']['tot'] - timers['start']['tot']))]
            logstr += ["mem={:.1f} GB".format(self.get_memory_usage_gb())]
            for key in timers['start'].keys():
                logstr += ["{:s}={:.1f} s".format(key, timers['end'][key] - timers['start'][key])]
            logstr += ["width={:d}".format(max_i_width)]
            self.logger.info(", ".join(logstr))


        #Finally resize matrix to match actually used observations
        if (store_full_matrix):
            M_indices.resize(nnz_counter)
            M_data.resize(nnz_counter)
            M_indptr = np.cumsum(M_indptr)

            self.logger.info("Reshaping M from {:d}x{:d} to {:d}x{:d} with {:d} nonzeros".format(total_num_obs, num_emissions, M_indptr.size-1, num_emissions, nnz_counter))
            self.M = scipy.sparse.csr_matrix((M_data, M_indices, M_indptr), shape=(M_indptr.size-1, num_emissions))


        self.y_0 = self.y_0[:obs_counter]
        self.sigma_o = self.sigma_o[:obs_counter]

        self.logger.debug("System matrix created.")





    #@profile
    def plotAshInvMatrix(self, fig=None, matrix=None, downsample=True):
        if (matrix is None):
            return Plot.plotAshInvMatrix(self.M, fig, downsample)
        else:
            return Plot.plotAshInvMatrix(matrix, fig, downsample)



    #@profile
    def save_matrices(self, outname, extra_data={}):
        """
        Save matrices to file.
        """
        self.logger.info("Writing matrices to {:s}".format(outname))
        tic = time.time()
        data =  {
            'Q': self.Q,
            'B': self.B,
            'y_0': self.y_0,
            'x_a': self.x_a,
            'sigma_o': self.sigma_o,
            'sigma_x': self.sigma_x,
            'ordering_index': self.ordering_index,
            'level_heights': self.level_heights,
            'volcano_altitude': self.volcano_altitude,
            'emission_times': self.emission_times
        }
        if hasattr(self, 'M'):
            self.logger.info("System size {:s}".format(str(self.M.shape)))
            data['M_data'] = self.M.data
            data['M_indices'] = self.M.indices
            data['M_indptr'] = self.M.indptr
            data['M_shape'] = self.M.shape
        data.update(extra_data)

        np.savez(outname, **data)
        toc = time.time()
        self.logger.info("Wrote to disk in {:.0f} s".format(toc-tic))



    #@profile
    def load_matrices(self, inname):
        """
        Load matrices from file
        """
        self.logger.info("Initializing from existing matrices")
        gc.collect()
        self.logger.info("Memory used pre load: {:.1f} GB".format(self.get_memory_usage_gb()))
        data = {}
        with np.load(inname) as npz_file:
            for key in npz_file.keys():
                data[key] = npz_file[key]

            self.B = data.pop('B')
            self.Q = data.pop('Q')
            self.y_0 = data.pop('y_0')
            self.x_a = data.pop('x_a')
            self.sigma_o = data.pop('sigma_o')
            self.sigma_x = data.pop('sigma_x')
            self.ordering_index = data.pop('ordering_index')
            self.level_heights = data.pop('level_heights')
            self.volcano_altitude = data.pop('volcano_altitude')
            self.emission_times = data.pop('emission_times')
            if 'M_data' in data.keys():
                self.logger.info("Loading full system matrix M")
                self.M = scipy.sparse.csr_matrix((data.pop('M_data'), data.pop('M_indices'), data.pop('M_indptr')), shape=data.pop('M_shape'))

        self.logger.info("Memory used post load: {:.1f} GB".format(self.get_memory_usage_gb()))
        self.logger.info("Ordering index size: {:s}".format(str(np.count_nonzero(self.ordering_index >= 0))))

        return data













    #@profile
    def iterative_inversion(self, solver='direct',
                            max_iter=100,
                            max_negative=1.0e-4,
                            a_priori_epsilon=1.0,
                            smoothing_epsilon=1.0e-3,
                            output_matrix_file=None,
                            output_fig_file=None,
                            output_fig_meta={}
                           ):
        """
        solver - Solver to use in the inversion procedure
        max_iter - Maximum number of iterations to perform
        max_negative - Ratio of negative to consider converged
        smoothing_epsilon - How smooth should the solution be?
        a_priori_epsilon - How close to the a priori should the solution be?
        output_matrix_file - Filename to store intermediate results to (numpy npz)
        output_fig_file - Filename to store intermediate results to (png file)

        returns x - a posteriori emissions
        """
        #3: Perform inversion based on description in Steensen PhD p30-32 (with corrections)
        self.logger.info("Starting iterative inversion")
        n_obs = self.y_0.size
        n_emis = self.x_a.size

        #Create second derivative matrix D
        D = np.zeros((n_emis, n_emis))
        np.fill_diagonal(D[1:], 1)
        np.fill_diagonal(D, -2)
        np.fill_diagonal(D[:,1:], 1)
        D[0, 0] = -1
        D[-1, -1] = -1
        DTD = np.matmul(D.transpose(), D)

        #These solvers do not change a priori uncertainty - no use in using multiple iterations
        if (solver=='nnls' or solver == 'lstsq' or a_priori_epsilon == 0.0):
            max_iter = 1

        converged = False
        self.convergence = []
        self.residual = []
        for i in range(max_iter):
            timings = { 'start': {}, 'end': {} }
            timings['start']['tot'] = time.time()
            timings['start']['G'] = time.time()
            self.logger.info("Iteration {:d}".format(i))
            self.logger.debug("|x_a|={:f}, |sigma_x|={:f}, |y_0|={:f}, |sigma_o|={:f}".format(np.linalg.norm(self.x_a), np.linalg.norm(self.sigma_x), np.linalg.norm(self.y_0), np.linalg.norm(self.sigma_o)))

            #Create diagonal matrix for sigma_x^-2
            if (a_priori_epsilon > 0):
                sigma_x_m2 = np.power(self.sigma_x / a_priori_epsilon, -2)
                sigma_x_m2_mean = np.mean(sigma_x_m2)
            else:
                sigma_x_m2 = np.zeros_like(self.sigma_x)
                sigma_x_m2_mean = 1.0

            #Create the system matrix G
            #   G = M^T diag(\sigma_o^-2) M + diag(\sigma_x^-2) + \eps D^T D
            G = self.Q + np.diag(sigma_x_m2) + smoothing_epsilon * sigma_x_m2_mean * DTD
            self.logger.debug("G condition number: {:f}".format(np.linalg.cond(G)))

            timings['end']['G'] = time.time()

            #Solve GX=B
            timings['start']['slv'] = time.time()
            self.logger.debug("Using solver {:s}".format(solver))
            if (solver=='direct'):
                #Uses a direct solver
                self.x = np.linalg.solve(G, self.B)
                res = np.linalg.norm(np.dot(G, self.x)-self.B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = self.x+self.x_a

            elif (solver=='inverse'):
                #Computes the inverse of G to solve Gx=B
                self.x = np.matmul(np.linalg.inv(G), self.B)
                res = np.linalg.norm(np.dot(G, self.x)-self.B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = self.x+self.x_a

            elif (solver=='pseudo_inverse'):
                #Computes the pseudo inverse of G to solve Gx=B
                self.x = np.matmul(scipy.linalg.pinv(G), self.B)
                res = np.linalg.norm(np.dot(G, self.x)-self.B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = self.x+self.x_a

            elif (solver=='lstsq'):
                #Use least squares to solve original M x = y_0 system (no a priori)
                assert hasattr(self, 'M'), "Full system matrix not computed - cannot use this solver!"
                self.x, res, rank, sing = np.linalg.lstsq(self.M.toarray(), self.y_0)
                self.logger.debug("Residuals: {:s}".format(str(res)))
                self.logger.debug("Singular values: {:s}".format(str(sing)))
                self.logger.debug("Rank: {:d}".format(rank))

            elif (solver=='lstsq2'):
                #Uses least squares to solve the least squares system... Should be equal direct solve?
                self.x, res, rank, sing = np.linalg.lstsq(G, self.B)
                self.x = self.x+self.x_a
                self.logger.debug("Residuals: {:s}".format(str(res)))
                self.logger.debug("Singular values: {:s}".format(str(sing)))
                self.logger.debug("Rank: {:d}".format(rank))

            elif (solver=='nnls'):
                assert hasattr(self, 'M'), "Full system matrix not computed - cannot use this solver!"
                #Uses non zero optimization to solve Mx=y (no a priori)
                self.x, rnorm = scipy.optimize.nnls(self.M.toarray(), self.y_0)
                self.logger.debug("Residual: {:f}".format(rnorm))

            elif (solver=='nnls2'):
                #Uses non zero optimization to solve Gx=B
                self.x, rnorm = scipy.optimize.nnls(G, self.B)
                self.x = self.x+self.x_a
                self.logger.debug("Residual: {:f}".format(rnorm))

            elif (solver=='lsq_linear'):
                assert hasattr(self, 'M'), "Full system matrix not computed - cannot use this solver!"
                #Linear least squares for Mx=y with a priori bounds
                res = scipy.optimize.lsq_linear(self.M.toarray(), self.y_0, bounds=(self.x_a-self.sigma_x/a_priori_epsilon, self.x_a+self.sigma_x/a_priori_epsilon))
                self.x = res.x
                self.logger.debug("Residuals: {:s}".format(str(res.fun)))
                self.logger.debug("Optimlity: {:f}".format(res.optimality))
                self.logger.debug("Iterations: {:d}".format(res.nit))
                self.logger.debug("Success: {:d}".format(res.success))

            elif (solver=='lsq_linear2'):
                #Linear least squares for Gx=B
                res = scipy.optimize.lsq_linear(G, self.B, bounds=(-self.sigma_x/a_priori_epsilon, +self.sigma_x/a_priori_epsilon))
                self.x = res.x+self.x_a
                self.logger.debug("Residuals: {:s}".format(str(res.fun)))
                self.logger.debug("Optimlity: {:f}".format(res.optimality))
                self.logger.debug("Iterations: {:d}".format(res.nit))
                self.logger.debug("Success: {:d}".format(res.success))

            else:
                raise RuntimeError("Invalid solver '{:s}'".format(solver))
            timings['end']['slv'] = time.time()


            #Write out statistics
            if hasattr(self, 'M'):
                res = np.linalg.norm(self.M.dot(self.x) - self.y_0)
                self.residual += [res]
                self.logger.info("Residual |M x - y_o| = {:f}".format(res))
            else:
                self.residual += [0.0]

            #Create plots and save result
            if (output_fig_file is not None):
                path, ext = os.path.splitext(output_fig_file)

                fig = self.plotAshInvMatrix(matrix=G)
                plt.suptitle("Iteration {:d}, residual={:f}, solver={:s}".format(i, res, solver))
                fig.savefig("{:s}_{:03d}_G{:s}".format(path, i, ext), metadata=output_fig_meta)
                plt.close()

            #Save matrices as npz file (useful for debugging)
            if (output_matrix_file is not None):
                path, ext = os.path.splitext(output_matrix_file)
                outname = "{:s}{:03d}{:s}".format(path, i, ext)
                self.logger.info("Saving matrix to {:s}".format(outname))
                np.savez_compressed(outname,
                             B=self.B,
                             G=G,
                             x_a=self.x_a,
                             sigma_x_m2=sigma_x_m2,
                             x=x)

            #Update sigma of emissions / a priori
            #Find indices of most negative elements
            sort_idx = np.argsort(self.x)
            pos_idx = sort_idx[self.x[sort_idx] >= 0]
            neg_idx = sort_idx[self.x[sort_idx] < 0]

            #Adjust at most 1/3 of the values in x
            neg_idx = neg_idx[:min(neg_idx.size, self.x.size//3)]

            #Pull solution towards a priori for negative values by decreasing a priori uncertainty
            self.sigma_x[neg_idx] = self.sigma_x[neg_idx] * 0.9
            #self.sigma_x[pos_idx] = np.minimum(self.sigma_x[pos_idx] * 1.1, self.sigma_x[pos_idx])

            #Check for convergence
            sum_positive = self.x[pos_idx].sum()
            sum_negative = self.x[neg_idx].sum()
            self.convergence += [-100*sum_negative/(sum_positive-sum_negative)]
            self.logger.info("Sum negative: {:f}, sum positive: {:f} ({:2.2f} %)".format(sum_negative, sum_positive, -100*sum_negative/(sum_positive-sum_negative)))

            timings['end']['tot'] = time.time()
            logstr = []
            for key in timings['start'].keys():
                logstr += ["{:s}={:.1f} s".format(key, timings['end'][key] - timings['start'][key])]
            self.logger.info("Timings: " + ", ".join(logstr))

            if (-sum_negative < (sum_positive-sum_negative) * max_negative):
                converged = True
                break

        if (converged):
            self.logger.info("Converged!")
        else:
            self.logger.warning("Solution did not converge!")





if __name__ == '__main__':
    import configargparse
    import sys
    parser = configargparse.ArgParser(description='Run iterative inversion algorithm.')
    parser.add("-c", "--config", is_config_file=True, help="config file which specifies options (commandline overrides)")
    parser.add("-v", "--verbose", help="Set output verbosity [0-100]", type=float, default=0)

    #Input arguments
    input_args = parser.add_argument_group('Input arguments')
    input_args.add("--matched_files", type=str, required=True,
                        help="CSV-file of matched files")
    input_args.add("--a_priori_file", type=str, required=True,
                        help="A priori emission json file")
    input_args.add("--input_unscaled", type=str, default=None,
                        help="Previously generated unscaled system matrices to run inversion on.")
    input_args.add("--progress_file", type=str, default=None,
                        help="Progress file to continue running an aborted run")
    input_args.add("--store_full_matrix", action="store_true",
                        help="Store the full M matrix (very slow - mostly for debugging)")
    input_args.add("--use_elevations", action='store_true',
                        help="Enable use of ash cloud elevations in inversion")

    #Output arguments
    output_args = parser.add_argument_group('Output arguments')
    output_args.add("--output_dir", type=str, required=True,
                        help="Output directory to place files in")

    #Dimension and scaling arguments
    scaling_args = parser.add_argument_group('System matrix dimension/scaling arguments')
    scaling_args.add("--scale_a_priori", type=float, default=1.0e-9,
                        help="Scaling factor for a priori estimate (e.g., kg=>Tg)")
    scaling_args.add("--scale_emission", type=float, default=1.0e-9,
                        help="Conversion factor from emission unit to kg (e.g, ug=>kg)")
    scaling_args.add("--scale_observation", type=float, default=1.0e-3,
                        help="Conversion factor from observation unit to kg (e.g., g=>kg)")

    #Solver agruments
    solver_args = parser.add_argument_group('Solver arguments')
    solver_args.add("--solver", type=str, default='direct',
                        help="Solver to use")
    solver_args.add("--a_priori_epsilon", type=float, action="append", default=[],
                        help="How much to weight a priori info")
    solver_args.add("--smoothing_epsilon", type=float, default=1.0e-1,
                        help="How smooth the solution should be")
    solver_args.add("--max_iter", type=int, default=20,
                        help="Maximum number of iterations to perform.")

    #Finally parse all arguments
    args = parser.parse_args()

    print("Arguments: ")
    print("=======================================")
    for var, val in vars(args).items():
        print("{:s} = {:s}".format(var, str(val)))
    print("=======================================")

    with open(args.config) as cfg_file:
        cfg = cfg_file.read()

    run_meta = {
        'arguments': json.dumps(vars(args)),
        'run_date': datetime.datetime.utcnow().isoformat(timespec='seconds'),
        'run_dir': os.getcwd(),
        'config': json.dumps(cfg)
    }


    if (args.verbose):
        logging.basicConfig(format='%(asctime)s %(levelname)s (%(name)s): %(message)s',
                        datefmt="%Y-%m-%dT%H:%M:%SZ",
                        stream=sys.stderr,
                        level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s (%(name)s): %(message)s',
                        datefmt="%Y-%m-%dT%H:%M:%SZ",
                        stream=sys.stderr,
                        level=logging.INFO)


    #Set output directory
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    output_basename = os.path.join(outdir, "inversion").rstrip("/")

    #Create inversion object
    ash_inv = AshInversion(verbose=args.verbose)

    #Use existing or create new unscaled file
    if (args.input_unscaled is not None):
        print("Loading existing matrices from {:s}".format(args.input_unscaled))
        ash_inv.load_matrices(args.input_unscaled)
    else:
        print("Creating new system matrices from {:s}".format(args.matched_files))
        unscaled_file = output_basename + "_system_matrix.npz"
        if (os.path.exists(unscaled_file)):
            print("System matrix {:s} already exists: please move or delete.".format(unscaled_file))
            sys.exit(-1)

        ash_inv.make_a_priori(args.a_priori_file, scale_a_priori=args.scale_a_priori)
        ash_inv.make_system_matrix(args.matched_files,
                                scale_emission=args.scale_emission,
                                scale_observation=args.scale_observation,
                                use_elevations=args.use_elevations,
                                progress_file=args.progress_file,
                                store_full_matrix=args.store_full_matrix,
                                )
        ash_inv.save_matrices(unscaled_file)

    #Run iterative inversion
    n_iter = len(args.a_priori_epsilon)
    use_bisection = False
    if (n_iter == 0):
        n_iter = 20
        interval = [0, 1]
        use_bisection = True
        print("Bisection to find optimal a priori epsilon")

    for i in range(n_iter):
        if (use_bisection):
            eps = (interval[0] + interval[1]) / 2
        else:
            eps = args.a_priori_epsilon[i]
        print("Inverting with a priori epsilon={:f}".format(eps))
        ash_inv.iterative_inversion(smoothing_epsilon=args.smoothing_epsilon,
                                a_priori_epsilon=eps,
                                max_iter=args.max_iter,
                                solver=args.solver)

        valid = ash_inv.x_a > 0
        l_inf = np.max(np.abs((ash_inv.x[valid]-ash_inv.x_a[valid]) / ash_inv.x_a[valid]))
        print("Eps is {:f}, residual is{:f}, convergence is {:s}".format(eps, ash_inv.residual[-1], str(ash_inv.convergence)))

        #Update interval
        if (use_bisection):
            if (l_inf < 1):
                #Maximum relative change less than 1
                interval[1] = eps
            else:
                #Converges "too slowly"
                interval[0] = eps
            print("New interval: {:s}".format(str(interval)))

        #Save inverted result as JSON file (similar to a priori)
        output_file = "{:s}_{:03d}_{:.8f}_a_posteriori.json".format(output_basename, i, eps)
        print("Writing output to {:s}".format(output_file))
        with open(output_file, 'w') as outfile:
            output = {
                'a_posteriori_2d': ash_inv.x,
                'a_priori_2d': ash_inv.x_a,
                'ordering_index': ash_inv.ordering_index,
                'volcano_altitude': ash_inv.volcano_altitude,
                'level_heights': ash_inv.level_heights,
                'emission_times': ash_inv.emission_times,
                'residual': ash_inv.residual,
                'convergence': ash_inv.convergence
            }
            output.update(run_meta)
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)
            json.dump(output, outfile, cls=NumpyEncoder)

    #Exit
    print("Inversion procedure complete - output in {:s}".format(args.output_dir))
