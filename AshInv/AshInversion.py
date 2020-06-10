#!/usr/bin/env python3
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
from netCDF4 import Dataset, num2date
import glob
import os
import re
import json
import scipy.linalg
import scipy.optimize
import gc
import psutil

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
            self.M = Misc.delete_in_place(self.M, to_keep_cols=to_keep)
        self.x_a = np.delete(self.x_a, to_delete)
        self.sigma_x = np.delete(self.sigma_x, to_delete)





    def make_a_priori(self, filename):
        """
        Create left hand side vector of emissions and emission uncertainties

        Read the a priori emission estimates from file
        and make the a priori vectors

        param: a_priori_file - CSV-file with dates and filenames for satellite observations
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





    def make_system_matrix(self,
                           matched_file_csv_filename,
                           use_elevations=False,
                           zero_thinning=0.7,
                           obs_zero=1.0e-20,
                           obs_flag_max=1.95):
        """
        Read the matched data and create the system matrix

        matched_file_csv_filename - filename of matched files (generated by matchfiles)
        use_elevations - Use elevation information (if available) in the inversion
        zero_thinning - Ratio of zero observations to keep
        obs_zero - Observations smaller than this are treated as zero
        obs_flag_max - Remove observations with an as flag higher than this (0=no ash, 1=ash, 2=don't know)"
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
        self.logger.debug("{:d} observations, {:d} emissions in {:d} files".format(total_num_obs, total_num_emis, num_files))

        #Check if we have observed altitudes
        #Then we double the number of observations: ash up to altitude, no ash above
        level_altitudes = np.cumsum(self.level_heights) + self.volcano_altitude
        with Dataset(os.path.join(matched_file_dir, matched_files_df.matched_file[0])) as nc_file:
            if (use_elevations and 'obs_alt' in nc_file.variables.keys()):
                self.logger.info("Using altitudes in inversion")
                total_num_obs *= 2
            else:
                use_elevations = False
                self.logger.info("Altitudes not used for inversion")

        def npTimeToDatetime(np_time):
            return datetime.datetime.utcfromtimestamp((np_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

        #Remove timesteps from a priori that don't match up with matched files (make system matrix far smaller)
        matched_times = np.array(matched_files_df["date"], dtype='datetime64[ns]')
        remove = np.ones((self.level_heights.size, self.emission_times.size), dtype=np.bool)
        for t, emission_time in enumerate(self.emission_times):
            min_days=0
            max_days=6,
            deltas = (matched_times - emission_time) / np.timedelta64(1, 'D')
            valid = (deltas >= min_days) & (deltas < max_days)
            if np.any(valid):
                remove[:,t] = False
        self.logger.info("Removing {:d}/{:d} a priori values without observations".format(np.count_nonzero(remove), self.emission_times.size*self.level_heights.size))
        self.make_ordering_index(remove)

        #Allocate data
        gc.collect()
        num_emissions = np.count_nonzero(self.ordering_index >= 0)
        assert num_emissions == (self.ordering_index.max()+1), "Ordering index has wrong entries!"
        self.logger.info("Memory available pre alloc: {:.1f} MB".format(psutil.virtual_memory().available/(1024*1024)))
        self.logger.info("Will try to allocate {:d}x{:d} matrix ({:.1f} MB)".format(total_num_obs, num_emissions, total_num_obs*num_emissions*8/(1024*1024)))
        self.M = np.zeros((total_num_obs, num_emissions), dtype=np.float64)
        self.y_0 = np.empty((total_num_obs), dtype=np.float64)
        self.sigma_o = np.empty((total_num_obs), dtype=np.float64)
        self.logger.info("Memory post alloc: {:.1f} MB".format(psutil.virtual_memory().available/(1024*1024)))

        #RHS vectors of observations (b) and uncertainties (b_sigma)
        obs_counter = 0

        #Loop over all observations and assemble row by row into matrix
        n_removed = 0
        start_assembly = time.time()
        for row in matched_files_df.itertuples():
            gc.collect()

            self.logger.debug("Processing {:s}\nCPU: {:.1f} %, Memory: {:.1f} %".format(row.matched_file, psutil.cpu_percent(1), psutil.virtual_memory().percent))

            filename = row.matched_file
            filename = os.path.join(matched_file_dir, filename)
            if (self.args['verbose'] > 50):
                self.logger.debug("Reading {:s}".format(row.matched_file))

            #Print progress every 10% of files
            if ((row.Index+1) % max(1, (num_files // 10)) == 0):
                percent_complete = (100*(obs_counter+n_removed+1)) / total_num_obs
                eta = (time.time()-start_assembly)/percent_complete*(100-percent_complete)
                self.logger.info("Processing file {:d}/{:d}: {:02.0f} % completed. ETA={:s}".format(
                    (row.Index+1),
                    num_files,
                    percent_complete,
                    str(datetime.timedelta(seconds=eta))))

            if(row.num_observations == 0):
                continue

            with Dataset(filename) as nc_file:
                #Get the indices for the time which corresponds to the a-priori data
                unit = nc_file['time'].units
                times = num2date(nc_file['time'][:], units = unit).astype('datetime64[ns]')
                #time_index = np.flatnonzero(np.isin(self.emission_times, time))
                time_index = np.full(times.shape, -1, dtype=np.int32)
                for i, t in enumerate(times):
                    ix = np.flatnonzero(self.emission_times == t)
                    if (len(ix) == 0):
                        self.logger.debug("Time {:s} is from {:s} does not match up with a priori emission!".format(str(t), filename))
                    elif (len(ix) == 1):
                        time_index[i] = ix
                    else:
                        self.logger.warning("Time {:s} exists multiple times in a priori - using first!".format(str(t)))
                        time_index[i] = ix[0]

                if (self.args['verbose'] > 70):
                    self.logger.debug("File times: {:s}, \na priori times: {:s}, \nindices: {:s}".format(str(times), str(self.emission_times), str(time_index)))
                if (self.args['verbose'] > 90):
                    self.logger.debug("Time indices: {:s}".format(str(time_index)))

                #Get observations that are unmasked
                obs = nc_file['obs'][:,:]
                mask = ~obs.mask
                obs = obs[mask]
                n_obs = len(obs)

                obs_flag = nc_file['obs_flag'][:,:]
                obs_flag = obs_flag[mask]

                if (use_elevations):
                    obs_alt = nc_file['obs_alt'][:,:]*1000 #FIXME: Given in KM
                    obs_alt = obs_alt[mask]

                #Read timestep by timestep in case of many timesteps (out of memory)
                sim = np.empty((len(times), row.num_altitudes, n_obs))
                for t in range(len(times)):
                    s = nc_file['sim'][t,:,:,:]
                    sim[t,:,:] = s[:,mask]

                #Check sizes
                assert (n_obs == row.num_observations), "Number of unmasked observations {:d} does not match up with expected {:d}".format(n_obs, row.num_observations)
                assert (row.num_timesteps == sim.shape[0]), "Number of timesteps {:d} does not match up to number of sims {:d}".format(row.num_timesteps, sim.shape[0])
                assert (row.num_altitudes == sim.shape[1]), "Number of altitudes {:d} does not match up to number of sims {:d}".format(row.num_altitudes, sim.shape[1])
                assert (row.num_altitudes == self.level_heights.size), "Number of altitudes does not match a priori!"
                if (self.args['verbose'] > 50):
                    self.logger.debug("#obs={:d} ({:d}/{:d}), #time={:d}, #alt={:d}".format(n_obs, obs_counter, total_num_obs, row.num_timesteps, row.num_altitudes))

                #Remove observations with high ash flag
                keep = (obs_flag < obs_flag_max)
                n_remove_af = keep.size - np.count_nonzero(keep)
                if (n_remove_af > 0):
                    obs = obs[keep]
                    obs_flag = obs_flag[keep]
                    if (use_elevations):
                        obs_alt = obs_alt[keep]
                    sim = sim[:,:,keep]
                    n_obs -= n_remove_af
                    n_removed += n_remove_af

                #Thinning observations of no ash observations:
                n_remove_zero = 0
                zeros = (obs <= obs_zero)
                n_total_zero = np.count_nonzero(zeros)
                if (zero_thinning < 1.0):
                    r = np.random.random(obs.shape)
                    remove = zeros & (r > zero_thinning)
                    n_remove_zero = np.count_nonzero(remove)
                    if (n_remove_zero > 0):
                        keep = ~remove
                        obs = obs[keep]
                        obs_flag = obs_flag[keep]
                        if (use_elevations):
                            obs_alt = obs_alt[keep]
                        sim = sim[:,:,keep]
                        n_obs -= n_remove_zero
                        n_removed += n_remove_zero
                self.logger.info("File {:d}/{:d}: Added {:d}/{:d} observations (removed {:d} uncertain, and {:d}/{:d} zeroes)".format(
                    (row.Index+1),
                    num_files,
                    n_obs,
                    row.num_observations,
                    n_remove_af,
                    n_remove_zero,
                    n_total_zero))

                #  Assemble system matrix
                for o in range(n_obs):
                    for a in range(row.num_altitudes):
                        for t in range(row.num_timesteps):
                            if time_index[t] >= 0:
                                val = sim[t, a, o]
                                emission_index = self.ordering_index[a, time_index[t]]
                                if emission_index >= 0 and not np.ma.is_masked(val):
                                    self.M[obs_counter, emission_index] = val

                    #FIXME Birthe uses constant standard deviation here.
                    self.y_0[obs_counter] = obs[o]
                    self.sigma_o[obs_counter] = self.y_0[obs_counter]*1.0
                    obs_counter = obs_counter+1

                    #This part implements top of ash cloud observation into the system
                    #by adding a zero-observation for the altitudes above the ash cloud
                    if (use_elevations):
                        # Find index of top of ash cloud observation
                        altitude_max = min(row.num_altitudes, np.searchsorted(level_altitudes, obs_alt[o])+1)

                        # Move simulation values
                        for a in range(altitude_max, row.num_altitudes):
                            for t in range(row.num_timesteps):
                                if time_index[t] >= 0:
                                    emission_index = self.ordering_index[a, time_index[t]]
                                    if (emission_index >= 0):
                                        self.M[obs_counter, emission_index] = self.M[obs_counter-1, emission_index]
                                        self.M[obs_counter-1, emission_index] = 0

                        #FIXME What should the standard deviation be here?
                        self.y_0[obs_counter] = 0.0 #Zero oserved ash here
                        self.sigma_o[obs_counter] = 0.0
                        obs_counter = obs_counter+1


        #Finally resize matrix to match actually used observations
        self.logger.info("Reshaping M from {:d} to {:d} rows".format(self.M.shape[0], obs_counter))
        rows_to_keep = np.zeros(self.M.shape[0], dtype=np.bool)
        rows_to_keep[:obs_counter] = True
        self.M = Misc.delete_in_place(self.M, rows_to_keep)
        self.y_0 = self.y_0[:obs_counter]
        self.sigma_o = self.sigma_o[:obs_counter]

        self.logger.debug("System matrix created.")


    def save_matrices(self, outname):
        """
        Save matrices to file.
        """
        self.logger.info("Writing matrices to {:s}".format(outname))
        self.logger.info("System size {:s}".format(str(self.M.shape)))
        np.savez_compressed(outname,
                     M=self.M,
                     y_0=self.y_0,
                     x_a=self.x_a,
                     sigma_o=self.sigma_o,
                     sigma_x=self.sigma_x,
                     ordering_index=self.ordering_index,
                     level_heights=self.level_heights,
                     volcano_altitude=self.volcano_altitude,
                     emission_times=self.emission_times)


    def plotAshInvMatrix(self, fig=None, matrix=None, downsample=True):
        if (matrix is None):
            return Plot.plotAshInvMatrix(self.M, fig, downsample)
        else:
            return Plot.plotAshInvMatrix(matrix, fig, downsample)



    def load_matrices(self, inname):
        """
        Load matrices from file
        """
        self.logger.info("Initializing from existing matrices")
        with np.load(inname, mmap_mode='r') as data:
            self.M = data['M'].copy()
            self.y_0 = data['y_0'].copy()
            self.x_a = data['x_a'].copy()
            self.sigma_o = data['sigma_o'].copy()
            self.sigma_x = data['sigma_x'].copy()
            self.ordering_index = data['ordering_index'].copy()
            self.level_heights = data['level_heights'].copy()
            self.volcano_altitude = data['volcano_altitude'].copy()
            self.emission_times = data['emission_times'].copy()




    def scale_system(self, scale_emission=1.0e-9,
                            scale_observation=1.0e-3,
                            scale_a_priori=1.0e0,
                            min_emission_uncertainty_scale=5.0e-2,
                            min_emission_scale=2.0e-2,
                            min_emission=2.0e-5,
                            min_emission_error=1.0e-5,
                            obs_zero=1.0e-20,
                            obs_model_error=1.0e-2,
                            obs_min_error=5.0e-3,
                            obs_zero_error_scale=1e2):
        """
        Scale problem to correct units - this is where a lot of magic sauce lies hidden...

        scale_emission - Scale emission to same unit as a priori/observation (typically ug => kg is 1e-9)
        scale_observation - Scale observation to same unit as a priori/emission (typically g => kg is 1e-3)
        scale_a_priori - Scale a_priori to same unit as emission/observation (typically kg => kg is 1)
        min_emission_uncertainty_scale - Set minimum uncertainty to average multiplied by scale
        min_emission_scale - Set minimum emission to average multiplied by scale", default=2e-2)
        obs_zero - Observations smaller than this are treated as zero (with higher error)
        obs_model_error - Error (sigma) of observations
        obs_min_error - Minimum error (sigma) of observations
        obs_zero_error_scale - Scaling of error for zero observations
        """
        self.logger.info("Scaling system matrix M to corresponding units")
        #Scale observation to correct unit
        self.y_0 *= scale_observation

        #Scale observation standard deviation to correct unit
        #Increase error for zero observations
        obs_min_error = np.where(self.y_0 < obs_zero, obs_min_error, obs_min_error*obs_zero_error_scale)
        self.sigma_o = self.sigma_o * scale_observation
        self.sigma_o = np.sqrt(self.sigma_o**2 + obs_model_error**2 + obs_min_error**2)

        #Scale system matrix to correct unit
        self.M *= scale_emission

        #Scale a priori and a priori standard deviation to correct unit
        x_a_min = np.maximum(self.x_a.mean(), min_emission)*min_emission_scale
        sigma_x_min = np.maximum(self.sigma_x.mean(), min_emission_error)*min_emission_uncertainty_scale
        self.x_a = np.maximum(self.x_a, x_a_min) * scale_a_priori
        self.sigma_x = np.maximum(self.sigma_x, sigma_x_min) * scale_a_priori



    def prune_system(self,
                     rowsum_threshold=None,
                     colsum_threshold=None,
                     a_priori_threshold=None
                    ):
        """
        Remove insignificant parts of the inversion system
        """
        #First remove all zero simulations (rows with only zeros)
        #If all simulations (columns) for a single observation (row) have zero emission,
        #they all match equally well/poorly
        if (rowsum_threshold is not None):
            nonzero_rows = np.sum(self.M, axis=1) > rowsum_threshold
            n_remove = nonzero_rows.size - np.count_nonzero(nonzero_rows)
            self.logger.info("Removing {:d} all zero rows from M".format(n_remove))
            if (n_remove > 0):
                self.M = Misc.delete_in_place(self.M, to_keep_rows=nonzero_rows)
                self.y_0 = self.y_0[nonzero_rows]
                self.sigma_o = self.sigma_o[nonzero_rows]

        #Then prune all zero observations (columns with only zeros)
        #If all observations (rows) for a single simulation (column) have zero emission,
        #they all match equally well/poorly
        if (colsum_threshold is not None):
            nonzero_cols = np.sum(self.M, axis=0) > colsum_threshold
            n_remove = nonzero_cols.size - np.count_nonzero(nonzero_cols)
            self.logger.info("Removing {:d} all zero columns from M".format(n_remove))
            if (n_remove > 0):
                to_remove = np.isin(self.ordering_index, np.nonzero(nonzero_cols), invert=True)
                self.make_ordering_index(to_remove)

        #Remove all zero emissions in a priori
        #This forces the solution to be zero in these points as well
        if (a_priori_threshold is not None):
            nonzero_cols = self.x_a > a_priori_threshold
            n_remove = nonzero_cols.size - np.count_nonzero(nonzero_cols)
            self.logger.info("Removing {:d} all zero values from a priori".format(n_remove))
            if (n_remove > 0):
                to_remove = np.isin(self.ordering_index, np.nonzero(nonzero_cols), invert=True)
                self.make_ordering_index(to_remove)





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
        self.logger.debug("Starting iterative inversion")
        n_obs, n_emis = self.M.shape

        assert n_obs == self.y_0.size, "y_0 has the wrong shape (should be number of observations, {:d}, but got {:d})".format(n_obs, self.y_0.size)
        assert n_emis == self.x_a.size, "x_a has the wrong shape (should be number of emissions, {:d}, but got {:d})".format(n_emis, self.x_a.size)

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
            self.logger.info("Iteration {:d}".format(i))
            self.logger.debug("|x_a|={:f}, |sigma_x|={:f}, |y_0|={:f}, |sigma_o|={:f}".format(np.linalg.norm(self.x_a), np.linalg.norm(self.sigma_x), np.linalg.norm(self.y_0), np.linalg.norm(self.sigma_o)))

            #Create diagonal matrices for sigma_o^-2 and sigma_x^-2
            sigma_o_m2 = np.power(self.sigma_o, -2)
            if (a_priori_epsilon > 0):
                sigma_x_m2 = np.power(self.sigma_x / a_priori_epsilon, -2)
            else:
                sigma_x_m2 = np.zeros_like(self.sigma_x)

            #Create the system matrix G
            #Compute M^T\diag(sigma_o^-2)
            MT_sigma_o = self.M.transpose().copy()
            for k in range(MT_sigma_o.shape[1]):
                MT_sigma_o[:,k] *= sigma_o_m2[k]
            #   G = M^T diag(\sigma_o^-2) M + diag(\sigma_x^-2) + \eps D^T D
            G = np.matmul(MT_sigma_o, self.M) \
                + np.diag(sigma_x_m2) \
                + smoothing_epsilon * np.mean(sigma_x_m2) * DTD
            #G = np.matmul(MT_sigma_o, self.M) \
            #    + np.diag(sigma_x_m2) \
            #    + smoothing_epsilon * np.mean(sigma_x_m2) * DTD
            self.logger.debug("G condition number: {:f}".format(np.linalg.cond(G)))

            #Create right hand side
            #Y = y_0 - M x_a
            Y = self.y_0 - np.matmul(self.M, self.x_a)
            #B = sigma_o^-2 M^T Y
            B = np.matmul(MT_sigma_o, Y)

            #Solve GX=B
            self.logger.debug("Using solver {:s}".format(solver))
            if (solver=='direct'):
                #Uses a direct solver
                self.x = np.linalg.solve(G, B)
                res = np.linalg.norm(np.dot(G, self.x)-B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = self.x+self.x_a

            elif (solver=='inverse'):
                #Computes the inverse of G to solve Gx=B
                self.x = np.matmul(np.linalg.inv(G), B)
                res = np.linalg.norm(np.dot(G, self.x)-B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = self.x+self.x_a

            elif (solver=='pseudo_inverse'):
                #Computes the pseudo inverse of G to solve Gx=B
                self.x = np.matmul(scipy.linalg.pinv(G), B)
                res = np.linalg.norm(np.dot(G, self.x)-B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = self.x+self.x_a

            elif (solver=='lstsq'):
                #Use least squares to solve original M x = y_0 system (no a priori)
                self.x, res, rank, sing = np.linalg.lstsq(self.M, self.y_0)
                self.logger.debug("Residuals: {:s}".format(str(res)))
                self.logger.debug("Singular values: {:s}".format(str(sing)))
                self.logger.debug("Rank: {:d}".format(rank))

            elif (solver=='lstsq2'):
                #Uses least squares to solve the least squares system... Should be equal direct solve?
                self.x, res, rank, sing = np.linalg.lstsq(G, B)
                self.x = self.x+self.x_a
                self.logger.debug("Residuals: {:s}".format(str(res)))
                self.logger.debug("Singular values: {:s}".format(str(sing)))
                self.logger.debug("Rank: {:d}".format(rank))

            elif (solver=='nnls'):
                #Uses non zero optimization to solve Mx=y (no a priori)
                self.x, rnorm = scipy.optimize.nnls(self.M, self.y_0)
                self.logger.debug("Residual: {:f}".format(rnorm))

            elif (solver=='nnls2'):
                #Uses non zero optimization to solve Gx=B
                self.x, rnorm = scipy.optimize.nnls(G, B)
                self.x = self.x+self.x_a
                self.logger.debug("Residual: {:f}".format(rnorm))

            elif (solver=='lsq_linear'):
                #Linear least squares for Mx=y with a priori bounds
                res = scipy.optimize.lsq_linear(self.M, self.y_0, bounds=(self.x_a-self.sigma_x/a_priori_epsilon, self.x_a+self.sigma_x/a_priori_epsilon))
                self.x = res.x
                self.logger.debug("Residuals: {:s}".format(str(res.fun)))
                self.logger.debug("Optimlity: {:f}".format(res.optimality))
                self.logger.debug("Iterations: {:d}".format(res.nit))
                self.logger.debug("Success: {:d}".format(res.success))

            elif (solver=='lsq_linear2'):
                #Linear least squares for Gx=B
                res = scipy.optimize.lsq_linear(G, B, bounds=(-self.sigma_x/a_priori_epsilon, +self.sigma_x/a_priori_epsilon))
                self.x = res.x+self.x_a
                self.logger.debug("Residuals: {:s}".format(str(res.fun)))
                self.logger.debug("Optimlity: {:f}".format(res.optimality))
                self.logger.debug("Iterations: {:d}".format(res.nit))
                self.logger.debug("Success: {:d}".format(res.success))

            elif (solver == 'experimental'):
                #Create the system matrix G
                #Compute M^T\diag(sigma_o^-2)
                MT_sigma_o = self.M.transpose().copy()
                for k in range(MT_sigma_o.shape[1]):
                    MT_sigma_o[:,k] *= sigma_o_m2[k]

                misfit_epsilon = 1.0 - a_priori_epsilon
                G = np.matmul(MT_sigma_o, self.M) \
                    + a_priori_epsilon/misfit_epsilon * np.diag(sigma_x_m2) \
                    + smoothing_epsilon/misfit_epsilon * np.mean(sigma_x_m2) * DTD
                self.logger.debug("G condition number: {:f}".format(np.linalg.cond(G)))

                #Create right hand side
                #B = sigma_o^-2 M^T y_o + sigma_x^-2 x_a
                B = np.matmul(MT_sigma_o, self.y_0) + sigma_x_m2*self.x_a

                #Solve
                self.x = np.linalg.solve(G, B)
                res = np.linalg.norm(np.dot(G, self.x)-B)
                self.logger.debug("Residual: {:f}".format(res))

            elif (solver == 'experimental2'):
                #Compute N = M x_a
                N = self.M.copy()
                assert(N.shape[1] == self.x_a.size)
                xx = self.x_a.copy()
                m = xx.min()
                r = xx.max()-m
                xx = (xx - m) / r
                for k in range(self.x_a.size):
                    N[:,k] *= xx[k]

                #Compute N^T\diag(sigma_o^-2)
                NT_sigma_o = N.transpose().copy()
                for k in range(sigma_o_m2.size):
                    NT_sigma_o[:,k] *= sigma_o_m2[k]

                #   G = N^T diag(\sigma_o^-2) N + diag(1/4) + \eps D^T D
                s = np.ones_like(xx)
                s = sigma_x_m2 * np.power(xx, 2)
                G = misfit_epsilon * np.matmul(NT_sigma_o, N) + a_priori_epsilon * np.diag(s)  + smoothing_epsilon * DTD
                self.logger.debug("G condition number: {:f}".format(np.linalg.cond(G)))

                #Create right hand side
                #B = sigma_o^-2 N^T y_0
                B = np.matmul(NT_sigma_o, self.y_0)

                #Solve
                self.x = np.linalg.solve(G, B)
                res = np.linalg.norm(np.dot(G, self.x)-B)
                self.logger.debug("Residual: {:f}".format(res))
                self.x = -(self.x * r + m)


            else:
                raise RuntimeError("Invalid solver '{:s}'".format(solver))

            #Write out statistics
            res = np.linalg.norm(np.matmul(self.M, self.x) - self.y_0)
            self.residual += [res]
            self.logger.info("Residual |M x - y_o| = {:f}".format(res))
            self.logger.debug("|M|={:f}, |M^t\sigma_o={:f} |G|={:f}".format(np.linalg.norm(self.M), np.linalg.norm(MT_sigma_o), np.linalg.norm(G)))
            self.logger.debug("|x|={:f}, |Y|={:f}, |B|={:f}".format(np.linalg.norm(self.x), np.linalg.norm(Y), np.linalg.norm(B)))

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
                             B=B,
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
    parser.add("-v", "--verbose", help="increase output verbosity", action="store_true")

    #Input arguments
    input_args = parser.add_argument_group('Input arguments')
    input_args.add("--matched_files", type=str, required=True,
                        help="CSV-file of matched files")
    input_args.add("--a_priori_file", type=str, required=True,
                        help="A priori emission json file")
    input_args.add("--input_unscaled", type=str, default=None,
                        help="Previously generated unscaled system matrices to run inversion on.")
    input_args.add("--use_elevations", action='store_true',
                        help="Enable use of ash cloud elevations in inversion")

    #Output arguments
    output_args = parser.add_argument_group('Output arguments')
    output_args.add("--output_dir", type=str, required=True,
                        help="Output directory to place files in")

    #Pruning arguments
    pruning_args = parser.add_argument_group('System matrix pruning arguments')
    pruning_args.add("--prune", action='store_true',
                        help="Enable pruning of system")
    pruning_args.add("--zero_thinning", type=float, default=0.7,
                        help="Fraction of zero observations to keep/remove")
    pruning_args.add("--obs_zero", type=float, default=1.0e-5,
                        help="Observations less than this treated as zero")
    pruning_args.add("--obs_flag_max", type=float, default=1.95,
                        help="Discard observations with an AF flag greater than this")
    pruning_args.add("--rowsum_threshold", type=float, default=1.e-5,
                        help="Remove rows (observations) with less than this row (simulation) sum")
    pruning_args.add("--colsum_threshold", type=float, default=1.e-5,
                        help="Remove columns (simulations) with less than this column (simulation) sum")
    pruning_args.add("--a_priori_threshold", type=float, default=None,
                        help="Remove a priori less than this from the inversion (force zero emission)")

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
    solver_args.add("--a_priori_epsilon", type=float, action="append", default=[0.5],
                        help="How much to weight a priori info")
    solver_args.add("--smoothing_epsilon", type=float, default=1.0e-3,
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
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

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

        #Recompute the a x_a a priori etc.
        ash_inv.make_a_priori(args.a_priori_file)
    else:
        print("Creating new system matrices from {:s}".format(args.matched_files))
        unscaled_file = output_basename + "_system_matrix.npz"
        if (os.path.exists(unscaled_file)):
            print("System matrix {:s} already exists: please move or delete.".format(unscaled_file))
            sys.exit(-1)

        ash_inv.make_a_priori(args.a_priori_file)
        ash_inv.make_system_matrix(args.matched_files,
                                zero_thinning=args.zero_thinning,
                                obs_zero=args.obs_zero,
                                obs_flag_max=args.obs_flag_max,
                                use_elevations=args.use_elevations
                                )
        ash_inv.save_matrices(unscaled_file)
        try:
            fig = ash_inv.plotAshInvMatrix()
            fig.savefig(output_basename + "_system_matrix_full.png", metadata=run_meta)
            plt.close()
        except MemoryError:
            self.logger.warning("Unable to plot matrix - out of memory")
            pass

    #Prune system matrix
    if (args.prune):
        print("Pruning system matrix")
        ash_inv.prune_system(
            rowsum_threshold=args.rowsum_threshold,
            colsum_threshold=args.colsum_threshold,
            a_priori_threshold=args.a_priori_threshold
        )
        if (ash_inv.M.size == 0):
            print("System matrix pruned to zero size (i.e., no observations match any of the simulations)... Exiting")
            sys.exit(-1)
        try:
            fig = ash_inv.plotAshInvMatrix()
            fig.savefig(output_basename + "_system_matrix_pruned.png", metadata=run_meta)
            plt.close()
        except MemoryError:
            self.logger.warning("Unable to plot matrix - out of memory")
            pass

    #Scale system matrix
    print("Scaling system matrix")
    ash_inv.scale_system(
        scale_a_priori=args.scale_a_priori,
        scale_emission=args.scale_emission,
        scale_observation=args.scale_observation
    )
    try:
        fig = ash_inv.plotAshInvMatrix()
        fig.savefig(output_basename + "_system_matrix_scaled.png", metadata=run_meta)
        plt.close()
    except MemoryError:
        self.logger.warning("Unable to plot matrix - out of memory")
        pass

    #Run iterative inversion
    print("Running iterative inversion")
    for eps in args.a_priori_epsilon:
        print("Inverting with a priori epsilon={:f}".format(eps))
        ash_inv.iterative_inversion(smoothing_epsilon=args.smoothing_epsilon,
                                a_priori_epsilon=eps,
                                max_iter=args.max_iter,
                                solver=args.solver)

        #Save inverted result as JSON file (similar to a priori)
        output_file = "{:s}_{:f}_a_posteriori.json".format(output_basename, eps)
        print("Writing output to {:s}".format(output_file))
        with open(output_file, 'w') as outfile:
            output = {
                'a_posteriori_2d': ash_inv.x.tolist(),
                'a_priori_2d': ash_inv.x_a.tolist(),
                'ordering_index': ash_inv.ordering_index.tolist(),
                'volcano_altitude': ash_inv.volcano_altitude,
                'level_heights': ash_inv.level_heights.tolist(),
                'emission_times': ash_inv.emission_times.tolist(),
                'residual': ash_inv.residual,
                'convergence': ash_inv.convergence
            }
            output.update(run_meta)
            json.dump(output, outfile)

    #Exit
    print("Inversion procedure complete - output in {:s}".format(args.output_dir))
