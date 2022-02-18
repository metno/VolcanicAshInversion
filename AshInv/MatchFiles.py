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
from netCDF4 import Dataset, num2date
import os
import re
import json
from joblib import Parallel, delayed
import multiprocessing

from AshInv import Misc


class MatchFiles:
    def __init__(self,
                 emep_runs,
                 satellite_observations,
                 emep_runs_basedir=None,
                 satellite_observations_basedir=None,
                 verbose=False):
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

        #Get simulation files
        self.sim_files = pd.read_csv(emep_runs, parse_dates=[0], comment='#')
        self.obs_files = pd.read_csv(satellite_observations, parse_dates=[0], comment='#')

        #Remove timezone information
        self.logger.warning("Converting times from CSV files to UTC")
        self.sim_files['date'] = self.sim_files['date'].dt.tz_convert(None)
        self.obs_files['date'] = self.obs_files['date'].dt.tz_convert(None)

        #Set path to be relative to csv file (if it does not exist already in relative path)
        if (emep_runs_basedir is not None):
            self.sim_files.filename = self.sim_files.filename.apply(lambda x: os.path.join(emep_runs_basedir, x))
        else:
            self.sim_files.filename = self.sim_files.filename.apply(lambda x: os.path.join(os.path.dirname(emep_runs), x))

        if (satellite_observations_basedir is not None):
            self.obs_files.filename = self.obs_files.filename.apply(lambda x: os.path.join(satellite_observations_basedir, x))
        else:
            self.obs_files.filename = self.obs_files.filename.apply(lambda x: os.path.join(os.path.dirname(satellite_observations), x))

        if (self.verbose):
            self.logger.debug("Simulation files: " + pprint.pformat(self.sim_files))
            self.logger.debug("Observation files: " + pprint.pformat(self.obs_files))


    def match_files(self, output_dir,
                    min_days=0, max_days=6,
                    mask_sim=True,
                    obs_flag_max=1.95,
                    zero_thinning=0.7,
                    obs_zero=1.0e-20,
                    dummy_observation_json=None,
                    fix_broken_netcdf_files=False,
                    volcano_lat=None,
                    volcano_lon=None,
                    max_distance=None):
        """
        Loops through all observation files, and tries to match observation with simulations

        Parameters
        ----------
        output_dir : path
            Directory to place output NetCDF files.
        mask_sim : bool, optional
            Mask the unused simulation data in output (saves space). The default is True.
        obs_flag_max - Remove observations with an as flag higher than this (0=no ash, 1=ash, 2=don't know)"
        zero_thinning - Ratio of zero observations to keep
        obs_zero - Observations smaller than this are treated as zero

        Returns
        -------
        None.

        """
        os.makedirs(output_dir, exist_ok=True)

        #Get first and last timestep in each simulation netcdf file to speed up processing
        def getNCTimes(row):
            try:
                nc_file = Dataset(row['filename'])

                sim_time = nc_file['time'][:]
                unit = nc_file['time'].units
                sim_time = num2date(sim_time, units = unit, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            except:
                raise RuntimeError("Invalid timezone info in {:s}".format(row['filename']))
            else:
                nc_file.close()

            #Raise error if timezone supplied in NetCDF file
            for i in range(len(sim_time)):
                if (sim_time[i].tzinfo is not None and sim_time[i].tzinfo is not datetime.timezone.utc):
                    raise RuntimeError("Invalid timezone info in {:s}".format(row['filename']))
                else:
                    #Remove timezone info
                    sim_time[i].replace(tzinfo=None)

            #Find minimum and maximum time
            min_time = sim_time[0]
            max_time = sim_time[0]
            for t in sim_time[1:]:
                min_time = min(min_time, t)
                max_time = max(max_time, t)

            return min_time, max_time

        self.sim_files['first_ts'], self.sim_files['last_ts'] = zip(*self.sim_files.apply(getNCTimes, axis=1))

        #Prune observation files that clearly do not match up
        valid_obs_files = np.empty(self.obs_files.shape[0], dtype=np.bool)
        for i in range(self.obs_files.shape[0]):
            #All observations with no matching simulation can be removed
            obs_time = self.obs_files.date[i]
            first = (obs_time - self.sim_files.first_ts) / datetime.timedelta(days=1)
            last = (obs_time - self.sim_files.last_ts) / datetime.timedelta(days=1)
            valid_sim_files = (first >= 0) & (last <= 0)
            valid_obs_files[i] = np.any(valid_sim_files)
        if (np.count_nonzero(valid_obs_files) < self.obs_files.shape[0]):
            self.logger.info("Removing {:d}/{:d} non-matching observation files".format(self.obs_files.shape[0]-np.count_nonzero(valid_obs_files), self.obs_files.shape[0]))
        self.obs_files = self.obs_files.iloc[valid_obs_files]
        self.obs_files.reset_index(drop=True, inplace=True)

        #Add extra columns to files to speed up processing in inversion
        self.obs_files["matched_file"] = None
        self.obs_files["num_observations"] = int(0)
        self.obs_files["num_timesteps"] = int(0)
        self.obs_files["num_altitudes"] = int(0)

        #Merge with existing matched csv file (to continue an existing matching with new observations)
        matched_out_csv = os.path.join(output_dir, "matched_files.csv")
        if (os.path.exists(matched_out_csv)):
            self.logger.info("Continuing existing run {:s}".format(matched_out_csv))
            processed_obs_files = pd.read_csv(matched_out_csv, parse_dates=[0])
            self.obs_files = pd.concat([self.obs_files, processed_obs_files])
            self.obs_files = self.obs_files.drop_duplicates('filename', keep='last')
            self.obs_files.reset_index(drop=True, inplace=True)
            
        def match_single_file(obs_file):
            out_filename = os.path.splitext(os.path.basename(obs_file.filename))[0] + "_matched.nc"
            out_filename_metadata = os.path.join(output_dir, os.path.splitext(out_filename)[0] + ".txt")
            
            if os.path.isfile(out_filename_metadata):
                try: 
                    with open(out_filename_metadata, 'r') as file:
                        data = file.read().splitlines() 
                        obs_file_index = int(data[0])
                        out_filename = data[1]
                        num_obs = int(data[2])
                        num_zero_obs = int(data[3])
                        num_timesteps = int(data[4])
                        num_altitudes = int(data[5])
                except:
                    self.logger.warning("File {:d}/{:d} metadata file failed - reprocessing - {:s}".format(obs_file.Index+1, self.obs_files.shape[0], out_filename))
                    pass
                else:
                    self.logger.info("File {:d}/{:d} already processed - {:s}".format(obs_file.Index+1, self.obs_files.shape[0], out_filename))
                    return [obs_file_index, out_filename, num_obs, num_zero_obs, num_timesteps, num_altitudes]
                    
            self.logger.info("Creating {:d}/{:d} - {:s}".format(obs_file.Index+1, self.obs_files.shape[0], out_filename))

            #Read observation data
            if (dummy_observation_json is not None):
                obs_lon, obs_lat, obs, obs_alt, obs_flag = self.make_dummy_observation_data(obs_file.date, dummy_observation_json)
            else:
                obs_lon, obs_lat, obs, obs_alt, obs_flag = self.read_observation_data(obs_file.filename, fix_broken_netcdf_files=fix_broken_netcdf_files, 
                                                                                        volcano_lat=volcano_lat, volcano_lon=volcano_lon, max_distance=max_distance)

            #Read simulation data
            try:
                sim_lon, sim_lat, sim_date, sim = self.read_simulation_data(obs_file.date, min_days=min_days, max_days=max_days)
            except ValueError as e:
                self.logger.error("Could not match {:s}. Got error {:s}".format(obs_file.filename, str(e)))
                return [obs_file.Index, obs_file.filename, 0, 0, 0, 0]

            #Make sure the two files match in size
            obs_lon, obs_lat, obs, obs_alt, obs_flag, sim = self.match_size(obs_lon, obs_lat, obs, obs_alt, obs_flag, sim_lon, sim_lat, sim)
            sim_lon = obs_lon
            sim_lat = obs_lat

            #Get valid observations
            mask = np.nonzero((obs_flag < 2))

            #Mask those elements we do not actually use
            obs = obs[mask[0], mask[1]].filled()
            if (obs_alt is not None):
                obs_alt = obs_alt[mask[0], mask[1]]
            obs_flag = obs_flag[mask[0], mask[1]]
            #n_sim_files, n_levels, lat, lon
            sim = sim[mask[0], mask[1], :, :]

            #Generate (lon, lat) coordinte for each pixel
            obs_lon, obs_lat = np.meshgrid(obs_lon, obs_lat)
            obs_lon = obs_lon[mask[0], mask[1]]
            obs_lat = obs_lat[mask[0], mask[1]]

            #Remove observations with high ash flag (uncertain)
            keep = (obs_flag < obs_flag_max)
            n_remove_af = keep.size - np.count_nonzero(keep)
            if (n_remove_af > 0):
                obs = obs[keep]
                obs_flag = obs_flag[keep]
                if (obs_alt is not None):
                    obs_alt = obs_alt[keep]
                sim = sim[keep, :, :]
                obs_lon = obs_lon[keep]
                obs_lat = obs_lat[keep]

            #Thinning observations of no ash observations:
            n_remove_zero = 0
            zeros = np.flatnonzero(obs <= obs_zero)
            n_total_zero = zeros.size
            if (zero_thinning > 0.0):
                r = np.random.random(zeros.shape)
                remove = zeros[r <= zero_thinning]
                n_remove_zero = remove.size
                if (n_remove_zero > 0):
                    keep = np.ones(obs.shape, dtype=np.bool)
                    keep[remove] = False
                    obs = obs[keep]
                    obs_flag = obs_flag[keep]
                    if (obs_alt is not None):
                        obs_alt = obs_alt[keep]
                    sim = sim[keep, :, :]
                    obs_lon = obs_lon[keep]
                    obs_lat = obs_lat[keep]
            self.logger.info("Added {:d} ash observations and {:d} no ash observations. Ignored {:d} uncertain, and {:d}/{:d} zeroes".format(
                obs.size-(n_total_zero-n_remove_zero),
                n_total_zero-n_remove_zero,
                n_remove_af,
                n_remove_zero,
                n_total_zero))
            num_zero_obs = n_total_zero-n_remove_zero

            #Write to file
            self.write_matched_data(obs_lon, obs_lat, obs, obs_alt, obs_flag, obs_file.date, sim, sim_date, os.path.join(output_dir, out_filename))
            
            #Write metadata-file
            statistics = [obs_file.Index, out_filename, obs.size, num_zero_obs, sim.shape[1], sim.shape[2]]
            with open(out_filename_metadata, 'w') as file:
                for stat in statistics:
                    file.write(str(stat) + "\n")
                    
            #Return statistics
            return statistics
        
        #Run file matching in parallel
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(match_single_file)(obs_file) for obs_file in self.obs_files.itertuples())

        total_num_obs = 0
        total_num_zero_obs = 0
        for result in results:
            #Update statistics for this file
            self.obs_files.at[result[0], "matched_file"] = result[1]
            self.obs_files.at[result[0], "num_observations"] = result[2]
            self.obs_files.at[result[0], "num_timesteps"] = result[4]
            self.obs_files.at[result[0], "num_altitudes"] = result[5]
            
            total_num_obs += result[2]
            total_num_zero_obs += result[3]

        #Update CSV-file
        self.obs_files.to_csv(matched_out_csv, index=False)
        self.logger.info("Added a total of {:d} observations ({:.2f} % zeros) from {:d} files".format(total_num_obs, 100*total_num_zero_obs/max(1, total_num_obs), len(results)))



    def match_size(self, obs_lon, obs_lat, obs, obs_alt, obs_flag, sim_lon, sim_lat, sim):
        """
        Makes sure the size of observation and simulation are equal.

        Parameters
        ----------
        obs_lon : 1D np.array
            Longitude of observations.
        obs_lat : 1D np.array
            Latitude of observations.
        obs : np.array
            Observations.
        obs_alt : np.array
            Observation (max) altitude.
        obs_flag : np.array
            Observation flag.
        sim_lon : 1D np.array
            Longitude of simulation.
        sim_lat : 1D np.array
            Latitude of simulation.
        sim : np.array
            Simulation data.

        Returns
        -------
        lon : 1D np.array
            Longitudes common to both sim and obs.
        lat : 1D np.array
            Latitudes common to both sim and obs.
        obs : np.array
            Observations.
        obs_flag : np.array
            Observation flags.
        sim : np.array
            Simulation data.

        """
        if (sim.shape[0:2] != obs.shape):
            self.logger.debug("Size of simulations ({:s}) and observation ({:s}) should match, cropping!".format(str(sim.shape), str(obs.shape)))

            #Get common lon/lat coordinates
            obs_lon_valid = list(map(lambda lon: lon in sim_lon, obs_lon))
            obs_lat_valid = list(map(lambda lat: lat in sim_lat, obs_lat))
            sim_lon_valid = list(map(lambda lon: lon in obs_lon, sim_lon))
            sim_lat_valid = list(map(lambda lat: lat in obs_lat, sim_lat))

            #Crop observation
            obs_lon = obs_lon[obs_lon_valid]
            obs_lat = obs_lat[obs_lat_valid]
            obs = obs[obs_lat_valid,:]
            obs = obs[:,obs_lon_valid]
            if (obs_alt is not None):
                obs_alt = obs_alt[obs_lat_valid,:]
                obs_alt = obs_alt[:,obs_lon_valid]
            obs_flag = obs_flag[obs_lat_valid,:]
            obs_flag = obs_flag[:,obs_lon_valid]

            #Crop simulation
            sim_lon = sim_lon[sim_lon_valid]
            sim_lat = sim_lat[sim_lat_valid]
            sim = sim[sim_lat_valid,:,:,:]
            sim = sim[:,sim_lon_valid,:,:]

        if (self.verbose):
            self.logger.debug("Size of simulations ({:s}) and observation ({:s})".format(str(sim.shape), str(obs.shape)))

        assert np.allclose(obs_lon, sim_lon), "Lon coordinates do not agree!"
        assert np.allclose(obs_lat, sim_lat), "Lat coordinates do not agree!"

        return obs_lon, obs_lat, obs, obs_alt, obs_flag, sim



    def write_matched_data(self, lon, lat, obs, obs_alt, obs_flag, obs_time, sim, sim_dates, out_filename):
        """
        Write the matched data to output file

        Parameters
        ----------
        lon : 1d np.array
            Longitudes.
        lat : 1d np.array
            Latitudes.
        obs : 2d np.array
            Observations.
        obs_alt : 2d np.array
            Observation (max) altitudes
        obs_flag : 2d np.array
            Observation flags.
        obs_time : datetime-like
            Time of observation.
        sim : 4d np.array
            Simulation data as [time', 'alt', 'lat', 'lon'].
        sim_dates : 1d np.array
            Dates of simulations.
        out_filename : str
            Filename to write to.

        Raises
        ------
        Netcdf errors
            If unable to write to NetCDF file.

        Returns
        -------
        None.

        """
        self.logger.debug("Writing matched file " + out_filename)

        n_obs = obs.size
        self.logger.debug("Number of observations: ({:s})".format(str(n_obs)))

        #Make sure that we have the correct size of
        assert n_obs == lon.size, str(lon.size)
        assert n_obs == lat.size, str(lat.size)
        assert n_obs == obs.size, str(obs.size)
        if (obs_alt is not None):
            assert n_obs == obs_alt.size, str(obs_alt.size)
        assert n_obs == obs_flag.size, str(obs_flag.size)
        assert n_obs == sim.shape[0], str(sim.shape[0])

        try:
            nc_file = Dataset(out_filename, 'w')

            nc_file.createDimension('nobs', n_obs)
            nc_file.createDimension('time', sim.shape[1])
            nc_file.createDimension('alt', sim.shape[2])

            nc_var = {}
            nc_var['time'] = nc_file.createVariable('time', 'f8', ('time',))

            nc_var['lon'] = nc_file.createVariable('longitude', 'f8', ('nobs',))
            nc_var['lat'] = nc_file.createVariable('latitude', 'f8', ('nobs',))
            nc_var['obs'] = nc_file.createVariable('obs', 'f8', ('nobs',), zlib=True, fill_value=-1.0)
            if (obs_alt is not None):
                nc_var['obs_alt'] = nc_file.createVariable('obs_alt', 'f8', ('nobs',), zlib=True, fill_value=-1.0)
            nc_var['obs_flag'] = nc_file.createVariable('obs_flag', 'f8', ('nobs',), zlib=True, fill_value=2) #2 == uncertain
            nc_var['sim'] = nc_file.createVariable('sim', 'f8', ('nobs', 'time', 'alt',), zlib=True, fill_value=-1.0)

            nc_var['lon'][:] = lon
            nc_var['lat'][:] = lat

            epoch = np.datetime64('1970-01-01T00:00')
            units = "days since 1970-01-01 00:00"
            nc_var['time'].units = units
            nc_var['time'][:] = [((t - epoch) / datetime.timedelta(days=1)) for t in sim_dates]

            nc_var['obs'][:] = obs
            nc_var['obs'].date_taken = str(obs_time)
            if (obs_alt is not None):
                nc_var['obs_alt'][:] = obs_alt
            nc_var['obs_flag'][:] = obs_flag
            nc_var['sim'][:,:,:] = sim
        except Exception as e:
            self.logger.error("Something went wrong:" + str(e))
            raise e
        finally:
            nc_file.close()


    def make_dummy_observation_data(self, obs_time,
                                    dummy_observation_json):
        """
        Makes a dummy observation based on the simulation

        Parameters
        ----------
        obs_time : datetime-like
            Time of observation to generate for.
        dummy_observation_json : A priori json file to use for making dummy observation
            Timesteps to use.

        Returns
        -------
        lon : TYPE
            DESCRIPTION.
        lat : TYPE
            DESCRIPTION.
        obs_alt : TYPE
            DESCRIPTION.
        obs : TYPE
            DESCRIPTION.
        obs_flag : TYPE
            DESCRIPTION.

        """


        #Open dummy a priori file to generate observation from
        with open(dummy_observation_json, 'r') as infile:
            #Read file
            tmp = json.load(infile)

            rand_threshold = 1.0 - tmp["observations_fraction"] #0.3 - Fraction of ash domain that we observe with our satellite
            rand_zero_threshold = 1.0 - tmp["zero_observations_fraction"] #0.05 - Fraction of no-ash domain that we observe with our satellite
            min_obs_threshold = tmp["min_obs_threshold"] #Observations less than this are considered zero
            emission_times = np.array(tmp["emission_times"], dtype='datetime64[ns]')
            level_heights = np.array(tmp["level_heights"], dtype=np.float64)
            volcano_altitude = tmp["volcano_altitude"]
            a_priori_2d = np.array(tmp["a_priori_2d"], dtype=np.float64) * 1.0e-9 #Kg to Tg
            altitude_scale = np.array(tmp["altitude_scale"], dtype=np.float64) # Scale observed altitude by some factor
            level_pattern = tmp["netcdf_level_pattern"]
            num_timesteps = emission_times.size
            num_altitudes = level_heights.size
            level_altitudes = volcano_altitude + np.cumsum(level_heights)*altitude_scale

        obs = None
        obs_alt = None
        lon = None
        lat = None

        #Prune invalid timesteps from a priori
        timesteps = np.zeros((num_timesteps), dtype=np.int32)
        for timestep in range(num_timesteps):
            t = emission_times[timestep]
            sim_times = np.array(self.sim_files.date, dtype='datetime64[ns]')
            t_index = np.argmin(np.abs(t - sim_times))

            #Find file matching emission time
            if ((t - sim_times[t_index]) / np.timedelta64(1, 'm') > 5):
                self.logger.warning("Could not find file matching time {:s}".format(str(t)))
                timesteps[timestep] = -1
                continue
            else:
                timesteps[timestep] = t_index

        #Loop over all files
        for timestep in timesteps:
            if (timestep < 0):
                continue

            filename = self.sim_files.filename[timestep]

            with Dataset(filename) as nc_file:
                #Read lat and lon
                if (lon is None):
                    lon = nc_file['lon'][:]
                if (lat is None):
                    lat = nc_file['lat'][:]

                #Find timestep closest to observation
                sim_time = num2date(nc_file['time'][:], units = nc_file['time'].units).astype('datetime64[ns]')
                nc_timestep_index = np.argmin(np.abs(sim_time - np.array(obs_time, dtype='datetime64[ns]')))

                #Get correct altitude - or all if None
                for altitude in range(num_altitudes):
                    var = level_pattern.format(altitude+1)
                    o = a_priori_2d[timestep, altitude] * nc_file[var][nc_timestep_index,:,:]
                    if (obs is None):
                        obs = np.zeros_like(o)
                        obs_alt = np.zeros_like(o)
                    obs += o
                    #Update altitude only if we have "certain ash" in this level
                    obs_alt[o > 1.0e-5] = np.maximum(obs_alt[o > 1.0e-5], level_altitudes[altitude] / 1000.0) #Altitude in km

        obs = obs * 1.0e-6 #ug to g

        #Remove random pixels
        #FIXME: Hard coded values here
        r = np.random.random(obs.shape)
        mask = ((obs > min_obs_threshold) & (r > rand_threshold)) | (r > rand_zero_threshold)
        obs = np.ma.masked_array(obs, ~mask)
        obs_alt = np.ma.masked_array(obs_alt, obs.mask)
        obs_flag = np.ma.masked_array(np.ones_like(obs), obs.mask)
        obs_flag[obs < 1.0e-5] = 2 #Uncertain if ash or not
        obs_flag[obs == 0] = 0     #Quite certain no ash

        return lon, lat, obs, obs_alt, obs_flag


    def read_observation_data(self, filename,
                 netcdf_observation_varnames=["AshMass", "ash_loading", "ash_concentration_col"],
                 netcdf_observation_altitude_varnames=["AshHeight"],
                 netcdf_observation_error_varnames=["AshFlag", "AF", "ash_flag"],
                 fix_broken_netcdf_files=False,
                 volcano_lat=None, 
                 volcano_lon=None, 
                 max_distance=None):
        """
        Reads the observation and observation flag from the NetCDF file

        Parameters
        ----------
        filename : str
            Filename to read.
        netcdf_observation_varnames : TYPE, optional
            DESCRIPTION. The default is ["AshMass", "ash_loading"].
        netcdf_observation_altitude_varnames : TYPE, optional
            DESCRIPTION. The default is ["AshHeight"].
        netcdf_observation_error_varnames : TYPE, optional
            DESCRIPTION. The default is ["AshFlag", "AF"].

        Returns
        -------
        lon : 1d np.array
            Longitudes.
        lat : 1d np.array
            Latitudes.
        obs : 2d np.array
            Observations.
        obs_alt : 2d np.array
            Observations (max) altitude
        obs_flag : 2d np.array
            Observation flag.

        """
        self.logger.debug("Reading observation file " + filename)

        # Get variable names
        netcdf_observation_varname = None
        netcdf_observation_altitude_varname = None
        netcdf_observation_error_varname = None

        with Dataset(filename, mode='r') as nc_file:
            nc_varnames = nc_file.variables.keys()
            for name in netcdf_observation_varnames:
                if name in nc_varnames:
                    netcdf_observation_varname = name
                    break

            for name in netcdf_observation_altitude_varnames:
                if name in nc_varnames:
                    netcdf_observation_altitude_varname = name
                    break

            for name in netcdf_observation_error_varnames:
                if name in nc_varnames:
                    netcdf_observation_error_varname = name
                    break

        assert netcdf_observation_varname is not None, "Could not get ash mass variable name"
        assert netcdf_observation_error_varnames is not None, "Could not get ash flag variable name"

        # Fix improper valid_min
        # Try 10 times if another process has the file open already
        if (fix_broken_netcdf_files):
            for i in range(10):
                try:
                    with Dataset(filename, mode='r+') as nc_file:
                        obs_var = nc_file[netcdf_observation_varname]
                        if ("valid_min" in obs_var.__dict__ and hasattr(obs_var.valid_min, '__len__') and obs_var.valid_min[-1] == "f"):
                            self.logger.warning("Observation file has wrong valid_min (not parseable as float...). Trying to fix")
                            obs_var.valid_min = obs_var.valid_min[:-1]
                except:
                    time.sleep(10)
                else:
                    break


        # Get actual data
        with Dataset(filename, mode='r') as nc_file:
            if ('longitude' in nc_file.variables.keys()):
                lon = nc_file['longitude'][:]
                lat = nc_file['latitude'][:]
            elif ('number_x_pixels' in nc_file.variables.keys()):
                lon = nc_file['number_x_pixels'][:]
                lat = nc_file['number_y_pixels'][:]
            else:
                raise RuntimeError("No supported spatial dimension")

            try:
                obs = nc_file[netcdf_observation_varname][:,:]
                if (netcdf_observation_altitude_varname is not None):
                    obs_alt = nc_file[netcdf_observation_altitude_varname][:,:]
                else:
                    obs_alt = None
                obs_flag = nc_file[netcdf_observation_error_varname][:,:]
            except Exception as e:
                self.logger.error("Error reading observations from {:s}, please check your NetCDF file".format(filename))
                raise e

        #Filter out distances that are too far away by marking them as 2 - uncertain
        if (max_distance is not None):
            m_lon, m_lat = np.meshgrid(lon, lat)
            distances = Misc.great_circle_distance_in_meters(volcano_lon, volcano_lat, m_lon, m_lat)/1000
            mask = distances > max_distance
            self.logger.info("Filtering out {:d}/{:d} observations {:f} km from lon={:f} lat={:f}".format(np.sum(obs_flag[mask]<2), np.sum(obs_flag<2), max_distance, volcano_lon, volcano_lat))
            obs_flag[mask] = 2

        #Possibly transpose observation
        if (obs.shape == (len(lon), len(lat))):
            obs = obs.T
            obs_flag = obs_flag.T

        assert obs.shape == obs_flag.shape, "Observation and observation flag must have same shape!"
        return lon, lat, obs, obs_alt, obs_flag





    def read_simulation_data(self, obs_time,
                             min_days=0, max_days=6,
                             level_regex="COLUMN_ASH_L(\d+)_k(\d+)"):
        """
        Reads all the simulation files in valid_sim_files, and returns an array
        data whose size is nx*ny*n_levels*n_sim_files

        Parameters
        ----------
        obs_time : datetime-like
            Time of obeservation.
        min_days : float, optional
            Minimum days from emission to observation. The default is 0.
        max_days : float, optional
            Maximum days from emission to observation. The default is 6.
        level_regex : str, optional
            Regex to match variable name in NetCDF simulatoin file. The default is "COLUMN_ASH_L(\d+)_k(\d+)".

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        lon : 1d np.array
            Longitudes.
        lat : 1d np.array
            Latitudes.
        date : 1d np.array
            Dates of emission times.
        data : 4d np.array
            Simulation data as nx*ny*n_levels*n_sim_files.

        """
        level_regex = re.compile(level_regex)

        #Find simulation files within reasonable time distance from observation, i.e., [0, 6] days
        #and process these
        first = (obs_time - self.sim_files.first_ts) / datetime.timedelta(days=1)
        last = (obs_time - self.sim_files.last_ts) / datetime.timedelta(days=1)
        valid_sim_files = (first >= min_days) & (last <= max_days)
        valid_sim_files = np.atleast_1d(np.squeeze(np.nonzero(valid_sim_files)))
        if (len(valid_sim_files) == 0):
            raise ValueError("No valid simulation files found!")

        #Our local data of the netcdf simulation file contents - initialize to NaN to make sure everything is written to
        n_valid = len(valid_sim_files)

        nx, ny, n_levels = None, None, None

        #Loop over simulation files (inner loop)
        non_matching_files=[]
        for out_i, in_i in enumerate(valid_sim_files):
            emission_time = self.sim_files.date[in_i]
            filename = self.sim_files.filename[in_i]

            #Initialize data sizes based on first file
            if (out_i == 0):
                with Dataset(filename) as nc_file:
                    lon = nc_file['lon'][:]
                    lat = nc_file['lat'][:]
                    nx = len(lon)
                    ny = len(lat)

                    n_levels = 0
                    for var in nc_file.variables:
                        varname = str(var)
                        r = level_regex.match(varname)
                        if (r):
                            n_levels += 1

                data = np.empty((ny, nx, n_valid, n_levels))
                if (self.verbose):
                    self.logger.debug("nx={:d}, ny={:d}, n_levels={:d}, n_valid={:d}".format(nx, ny, n_levels, n_valid))

            #Read data from file
            with Dataset(filename) as nc_file:
                sim_nx = len(nc_file['lon'])
                sim_ny = len(nc_file['lat'])

                assert sim_nx == nx, "All simulations must have the same dimensions"
                assert sim_ny == ny, "All simulations must have the same dimensions"

                #Find timestep closest to observation
                sim_time = nc_file['time'][:]
                unit = nc_file['time'].units
                sim_time = num2date(sim_time, units=unit, only_use_cftime_datetimes=False, only_use_python_datetimes=True)

                for i in range(len(sim_time)):
                    if (sim_time[i].tzinfo is not None and sim_time[i].tzinfo is not datetime.timezone.utc):
                        raise RuntimeError("Invalid timezone info in {:s}".format(filename))
                    else:
                        #Remove timezone info and convert to numpy time
                        sim_time[i].replace(tzinfo=None)
                sim_time = sim_time.astype(np.datetime64)

                #FIXME: hour is defined as :30 (average), hourInst as :00. Which to choose here?
                timestep = np.argmin(np.abs(sim_time - np.array(obs_time, dtype=np.datetime64)))

                if (self.verbose):
                    self.logger.debug("Observation time {:s}, matching emission time: {:s} ({:s}, timestep {:d} - {:s})".format(str(obs_time), str(emission_time), filename, timestep, str(sim_time[timestep])))
                #FIXME: What is a reasonable delta here?
                if (np.abs(sim_time[timestep] - np.array(obs_time, dtype=np.datetime64)) > datetime.timedelta(minutes=30)):
                    non_matching_files += [os.path.basename(filename)]
                    continue

                n_levels_read = 0
                for var in nc_file.variables:
                    varname = str(var)
                    r = level_regex.match(varname)
                    if (r):
                        assert len(r.groups()) == 2, "Something wrong with regex matching"
                        # The levels are numbered from 1
                        level = int(r.group(1)) - 1
                        data[:, :, out_i, level] = nc_file[varname][timestep,:,:]
                        n_levels_read += 1
                        #if (self.verbose):
                        #    self.logger.debug("Wrote {:d}-{:d} from {:s} (sum={:f})".format(out_i, level, sim_file.filename, np.sum(data[out_i, level, :, :])))
                if (n_levels_read != n_levels):
                    if (self.verbose):
                        self.logger.warning("Number of expected ({:d}) and found ({:d}) levels did not match!".format(n_levels, n_levels_read))

        date = self.sim_files.date[valid_sim_files]

        if (len(non_matching_files) > 0):
            if (self.verbose):
                self.logger.warning("No matching timestep for observation time {:s} in {:s}".format(str(obs_time), " ".join(non_matching_files)))
            else:
                self.logger.warning("No matching timestep for observation time {:s} in {:d} files".format(str(obs_time), len(non_matching_files)))
        return lon, lat, date, data








if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgParser(description='Match observations and simulations.')
    parser.add("--verbose", action="store_true", help="Enable verbose mode")
    parser.add("-c", "--config", is_config_file=True, help="config file which specifies options (commandline overrides)")
    parser.add('--simulation_csv', type=str,
                        help='CSV-file with EEMEP simulation runs', required=True)
    parser.add('--simulation_basedir', type=str,
                        help='Absolute path to where simulation files are', required=False, default=None)
    parser.add('--observation_csv', type=str,
                        help="CSV-file with satellite observations", required=True)
    parser.add('--observation_basedir', type=str,
                        help='Absolute path to where observation files are', required=False, default=None)
    parser.add("--output_dir", type=str,
                        help="Output dir to place matched data into", required=True)
    parser.add("--no_mask_sim", action="store_false",
                        help="Do not mask simulation data in output matched files (requires more storage - used for debugging)")
    parser.add("--obs_flag_max", type=float, default=1.95,
                        help="Discard observations with an AF flag greater than this")
    parser.add("--zero_thinning", type=float, default=0.25,
                        help="Fraction of zero observations to keep")
    parser.add("--obs_zero", type=float, default=1.0e-5,
                        help="Observations less than this treated as zero")
    parser.add("--dummy_observation_json", type=str,
                        help="JSON-file used to generate dummy observation", default=None)
    parser.add("--fix_broken_netcdf_files", action="store_true",
                        help="Fix (by overwriting) broken netcdf files.")
    parser.add("--random_seed", type=int,
                        help="Random seed (to make run reproducible)", default=0)
    parser.add('--max_days', type=float,
                        help="Maximum number of days between emission (simulation) and observation", default=6.0)
    parser.add('--min_days', type=float,
                        help="Minimum number of days between emission (simulation) and observation", default=0.0)
    parser.add('--volcano_lat', type=float,
                        help="Latitude of volcano (used as distance criteria)", default=None)
    parser.add('--volcano_lon', type=float,
                        help="Longitude of volcano (used as distance criteria)", default=None)
    parser.add('--max_distance', type=float,
                        help="Maximum distance from volcano to consider (in km)", default=None)

    args = parser.parse_args()

    print("Arguments: ")
    for var, val in vars(args).items():
        print("{:s} = {:s}".format(var, str(val)))

    if (args.verbose):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    np.random.seed(seed=args.random_seed)

    match = MatchFiles(emep_runs=args.simulation_csv,
                   satellite_observations=args.observation_csv,
                   emep_runs_basedir=args.simulation_basedir,
                   satellite_observations_basedir=args.observation_basedir,
                   verbose=args.verbose)
    match.match_files(output_dir=args.output_dir,
                      mask_sim=args.no_mask_sim,
                      obs_flag_max=args.obs_flag_max,
                      zero_thinning=args.zero_thinning,
                      obs_zero=args.obs_zero,
                      dummy_observation_json=args.dummy_observation_json,
                      fix_broken_netcdf_files=args.fix_broken_netcdf_files,
                      max_days=args.max_days,
                      min_days=args.min_days,
                      volcano_lat=args.volcano_lat,
                      volcano_lon=args.volcano_lon,
                      max_distance=args.max_distance)
