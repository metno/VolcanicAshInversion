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


def meters_to_hPa(meters):
    hPa = np.power(1.0-meters*0.0065/288.15, 5.255)*1013.25
    return hPa

def hPa_to_meters(hPa):
    meters = (288.15 / 0.0065) * (1 - np.power(hPa / 1013.25, 1/5.255))
    return meters

def hybrid_to_hPa(P0, hya, hyb):
    hPa = P0 * hyb + hya
    return hPa

def hybrid_to_meters(P0, hya, hyb):
    meters = hPa_to_meters(hybrid_to_hPa(P0, hya, hyb))
    return meters

class APriori():
    def __init__(self, volcano_altitude,
                 particle_density,
                 fine_ash_fraction,
                 a_priori_uncertainty,
                 hybrid_levels_file,
                 num_emission_levels,
                 plume_heights_file,
                 a_priori_file,
                 verbose=False):
        self.logger = logging.getLogger(__name__)

        # Read in plume heights and time evolution of eruption
        self.vertical_levels = pd.read_csv(hybrid_levels_file,
                         sep='\s+',
                         skiprows=1,
                         header=None)
        self.vertical_levels.columns = ["idx", "hya", "hyb"]

        #Calculate level boundaries in approximate meters
        P0 = meters_to_hPa(volcano_altitude)
        level_boundaries = hybrid_to_meters(P0, self.vertical_levels["hya"].to_numpy()/100.0, self.vertical_levels["hyb"].to_numpy())
        level_boundaries = level_boundaries[::-1]
        self.logger.debug("Level boundaries {:s}".format(str(level_boundaries)))

        #Use only specific levels
        n_levels = num_emission_levels
        level_boundaries = level_boundaries[:n_levels+1]
        level_heights = np.diff(level_boundaries)

        self.logger.debug("Using level boundaries {:s}".format(str(level_boundaries)))
        self.logger.debug("Level heights {:s}".format(str(level_heights)))

        # Read in plume heights and time evolution of eruption
        self.plume_heights = pd.read_csv(plume_heights_file,
                         sep='\s+',
                         skiprows=1,
                         header=None,
                         parse_dates=[[0, 1], [2, 3]])
        self.plume_heights.columns = ["start", "end", "plume_height"]


        # Calculate the mass eruption rate from Mastin (2009) formula
        # i kg / s
        rho = particle_density
        h0 = volcano_altitude
        plume_heights = self.plume_heights["plume_height"].values * 1000
        h = ((plume_heights - h0)/1000) / 2
        eruption_rate = np.zeros_like(h)
        for i in range(len(h)):
            if (h[i] > 0):
                eruption_rate[i] = rho*h[i]**(1.0/0.241)

        # Calculate the duration of each measurement in seconds
        duration_df = (self.plume_heights["end"] - self.plume_heights["start"]) / datetime.timedelta(seconds=1)
        duration = duration_df.values


        # Distribute the ash emitted at time period t - t+1 into the vertical boxes up to the observed ash plume height
        n_times = len(duration)
        a_priori_2d = np.zeros((n_times, n_levels))
        a_priori_2d_uncertainty = np.zeros_like(a_priori_2d)
        for i in range(n_times):
            if (plume_heights[i] > 0):
                #mass = Rate [kg/s] * duration [s] * fine_fraction
                fine_ash_mass = eruption_rate[i] * duration[i] * fine_ash_fraction

                #Distribute the plume in the elevation up to the observed height
                plume_fill = np.minimum(np.maximum(plume_heights[i]-level_boundaries[:-1], 0), level_heights)
                ash_distribution = plume_fill / plume_heights[i]
                a_priori_2d[i,:] = fine_ash_mass * ash_distribution
                a_priori_2d_uncertainty[i,:] = a_priori_2d[i,:] * a_priori_uncertainty
                if (verbose):
                    self.logger.debug("Ash distribution: {:s}".format(str(ash_distribution)))

        #a_priori_2d[:,:] = a_priori_2d[:,:]*2.0
        #a_priori_2d = a_priori_2d * (1.0 + 0.25 * (np.random.random(a_priori_2d.shape) - 0.5))
        #a_priori_2d[:8,:] *= 0.25
        #a_priori_2d[8:,:] *= 2.0

        # Write A PRIORI file
        self.logger.debug("Writing output to {:s}".format(a_priori_file))
        with open(a_priori_file, 'w') as outfile:
            output = {
                'volcano_altitude': volcano_altitude,
                'hybrid_levels_file': hybrid_levels_file,
                'particle_density': particle_density,
                'fine_ash_fraction': fine_ash_fraction,
                'level_heights': level_heights.tolist(),
                'emission_times': [str(t) for t in self.plume_heights["start"].values],
                'a_priori_2d': a_priori_2d.tolist(),
                'a_priori_2d_uncertainty': a_priori_2d_uncertainty.tolist()
            }
            json.dump(output, outfile, indent=4)


if __name__ == "__main__":
    import configargparse

    parser = configargparse.ArgParser(description='Create a priori emission estimate.')
    parser.add("-c", "--config", is_config_file=True, help="config file which specifies options (commandline overrides)")
    parser.add("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add("--volcano_altitude", type=int, required=True, help="Volcano summit elevation (meters above sea level)")
    parser.add("--particle_density", type=float, default=2.75e3, help="Particle density (kg/m^3)")
    parser.add("--fine_ash_fraction", type=float, default=0.1, help="Fine ash fraction (m63 from Mastins relationship)")
    parser.add("--a_priori_uncertainty", type=float, default=0.5, help="A priori uncertainty (fraction of a priori value)")
    parser.add("--hybrid_levels_file", type=str, required=True, help="Vertical levels file")
    parser.add("--num_emission_levels", type=int, default=19, help="Number of emission levels to use (must match simulation)")
    parser.add("--plume_heights_file", type=str, required=True, help="Plume heights input file")
    parser.add("--a_priori_file", type=str, required=True, help="Output a priori file")
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

    a_priori = APriori(args.volcano_altitude,
                 args.particle_density,
                 args.fine_ash_fraction,
                 args.a_priori_uncertainty,
                 args.hybrid_levels_file,
                 args.num_emission_levels,
                 args.plume_heights_file,
                 args.a_priori_file,
                 args.verbose)
