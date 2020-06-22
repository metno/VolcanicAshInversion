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


import os
import numpy as np
import pandas as pd
import datetime
import logging
from netCDF4 import Dataset
import json

from AshInv import Misc

def aPrioriFromPlumeHeights(args,
                a_priori_uncertainty,
                plume_heights_file,
                verbose=False):
    logger = logging.getLogger(__name__)

    logger.debug("Using level boundaries {:s}".format(str(args['level_boundaries'])))
    logger.debug("Level heights {:s}".format(str(args['level_heights'])))

    # Read in plume heights and time evolution of eruption
    plume_heights_df = pd.read_csv(plume_heights_file,
                        sep='\s+',
                        skiprows=1,
                        header=None,
                        parse_dates=[[0, 1], [2, 3]])
    plume_heights_df.columns = ["start", "end", "plume_height"]

    # Calculate the mass eruption rate from Mastin (2009) formula
    # i kg / s
    rho = args['particle_density']
    h0 = args['volcano_altitude']
    plume_heights = plume_heights_df["plume_height"].values * 1000
    h = ((plume_heights - h0)/1000) / 2
    eruption_rate = np.zeros_like(h)
    for i in range(len(h)):
        if (h[i] > 0):
            eruption_rate[i] = rho*h[i]**(1.0/0.241)

    # Calculate the duration of each measurement in seconds
    duration_df = (plume_heights_df["end"] - plume_heights_df["start"]) / datetime.timedelta(seconds=1)
    duration = duration_df.values


    # Distribute the ash emitted at time period t - t+1 into the vertical boxes up to the observed ash plume height
    n_times = len(duration)
    a_priori_2d = np.zeros((n_times, args['level_heights'].shape[0]))
    a_priori_2d_uncertainty = np.zeros_like(a_priori_2d)
    for i in range(n_times):
        if (plume_heights[i] > 0):
            #mass = Rate [kg/s] * duration [s] * fine_fraction
            fine_ash_mass = eruption_rate[i] * duration[i] * args['fine_ash_fraction']

            #Distribute the plume in the elevation up to the observed height
            plume_fill = np.minimum(np.maximum(plume_heights[i]-args['level_boundaries'][:-1], 0), args['level_heights'])
            ash_distribution = plume_fill / plume_heights[i]
            a_priori_2d[i,:] = fine_ash_mass * ash_distribution
            a_priori_2d_uncertainty[i,:] = a_priori_2d[i,:] * a_priori_uncertainty
            if (verbose):
                logger.debug("Ash distribution: {:s}".format(str(ash_distribution)))

    #a_priori_2d[:,:] = a_priori_2d[:,:]*2.0
    #a_priori_2d = a_priori_2d * (1.0 + 0.25 * (np.random.random(a_priori_2d.shape) - 0.5))
    #a_priori_2d[:8,:] *= 0.25
    #a_priori_2d[8:,:] *= 2.0

    # Return a priori data
    return {
        'emission_times': [str(t) for t in plume_heights_df["start"].values],
        'a_priori_2d': a_priori_2d,
        'a_priori_2d_uncertainty': a_priori_2d_uncertainty
    }

def aPrioriFromDatFile(args, a_priori_file):
    #Read levels of a priori
    df = pd.read_csv(a_priori_file,
                sep='\s+',
                nrows=2,
                header=None)
    level_boundaries_orig = df.to_numpy()
    level_boundaries_orig = np.hstack((level_boundaries_orig[0,:], level_boundaries_orig[1,-1]))

    #Read a priori and uncertainty
    dateparse = lambda dt, tm: pd.datetime.strptime("{:s}T{:s}Z".format(dt, tm), '%Y%m%dT%H%M%SZ')
    df = pd.read_csv(a_priori_file,
                sep='\s+',
                skiprows=2,
                usecols=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                header=None,
                parse_dates={'datetime': [0, 1]},
                date_parser=dateparse)

    a_priori_df=df[1::2]
    a_priori_uncertainty_df=df[0::2]

    #Compute duration
    dates = a_priori_df['datetime'].to_numpy()
    durations = np.diff(dates) / np.timedelta64(1, 's')
    assert np.all(durations == durations[0])
    duration = durations[0]

    #Convert a priori from kg/s total mass to kg fine ash
    a_priori_orig = a_priori_df.to_numpy()[:,1:].astype(np.float64) * duration * args['fine_ash_fraction']
    a_priori_uncertainty_orig = a_priori_uncertainty_df.to_numpy()[:,1:].astype(np.float64) * duration * args['fine_ash_fraction']

    #Redistribute the a priori to the new levels
    a_priori = Misc.resample_2D(a_priori_orig, level_boundaries_orig, args['level_boundaries'])
    a_priori_uncertainty = Misc.resample_2D(a_priori_uncertainty_orig, level_boundaries_orig, args['level_boundaries'])

    return {
        'emission_times': [str(t) for t in dates],
        'a_priori_2d': a_priori,
        'a_priori_2d_uncertainty': a_priori_uncertainty,
        'level_boundaries_orig': level_boundaries_orig,
        'a_priori_2d_orig': a_priori_orig,
        'a_priori_2d_uncertainty_orig': a_priori_uncertainty_orig
    }


if __name__ == "__main__":
    import configargparse
    logger = logging.getLogger("Main")

    parser = configargparse.ArgParser(description='Create a priori emission estimate.')
    parser.add("-c", "--config", is_config_file=True, help="config file which specifies options (commandline overrides)")
    parser.add("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add("--volcano_altitude", type=int, required=True, help="Volcano summit elevation (meters above sea level)")
    parser.add("--particle_density", type=float, default=2.75e3, help="Particle density (kg/m^3)")
    parser.add("--fine_ash_fraction", type=float, default=0.1, help="Fine ash fraction (m63 from Mastins relationship)")
    parser.add("--a_priori_uncertainty", type=float, default=0.5, help="A priori uncertainty (fraction of a priori value)")
    parser.add("--hybrid_levels_file", type=str, required=True, help="Vertical levels file")
    parser.add("--num_emission_levels", type=int, default=19, help="Number of emission levels to use (must match simulation)")
    parser.add("--a_priori_file", type=str, required=True, help="Output a priori file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add("--plume_heights_file", type=str, help="Plume heights input file")
    group.add("--dat_file", type=str, help="Existing a priori dat file")
    args = parser.parse_args()

    print("Arguments: ")
    print("=======================================")
    for var, val in vars(args).items():
        print("{:s} = {:s}".format(var, str(val)))
    print("=======================================")

    if (args.verbose):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read in hybrid vertical levels from file
    vertical_levels = pd.read_csv(args.hybrid_levels_file,
                        sep='\s+',
                        skiprows=1,
                        header=None)
    vertical_levels.columns = ["idx", "hya", "hyb"]

    #Calculate level boundaries in approximate meters
    P0 = Misc.meters_to_hPa(args.volcano_altitude)
    level_boundaries = Misc.hybrid_to_meters(P0, vertical_levels["hya"].to_numpy()/100.0, vertical_levels["hyb"].to_numpy())
    level_boundaries = level_boundaries[::-1]
    level_boundaries = level_boundaries[:args.num_emission_levels+1]
    level_heights = np.diff(level_boundaries)
    logger.debug("Output level boundaries {:s}".format(str(level_boundaries)))
    logger.debug("Output level heights {:s}".format(str(level_heights)))

    output = {
            'level_boundaries': level_boundaries,
            'level_heights': level_heights,
            'volcano_altitude': args.volcano_altitude,
            'hybrid_levels_file': args.hybrid_levels_file,
            'particle_density': args.particle_density,
            'fine_ash_fraction': args.fine_ash_fraction,
        }

    if (args.plume_heights_file is not None):
        a_priori = aPrioriFromPlumeHeights(output,
                    args.a_priori_uncertainty,
                    args.plume_heights_file,
                    args.verbose)
        output.update(a_priori)
    elif (args.dat_file  is not None):
        a_priori = aPrioriFromDatFile(output, args.dat_file)
        output.update(a_priori)
    else:
        logger.error("No valid output!")

    logger.debug("Writing output to {:s}".format(args.a_priori_file))
    for key in output.keys():
        if isinstance(output[key], (np.ndarray, np.generic)):
            output[key] = output[key].tolist()
    with open(args.a_priori_file, 'w') as outfile:
        json.dump(output, outfile, indent=4)
