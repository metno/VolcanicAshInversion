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
import numpy.matlib
import re
from netCDF4 import Dataset, num2date
from io import StringIO
import pandas as pd

from scipy.integrate import quad
from scipy.interpolate import PPoly

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def meters_to_hPa(meters, P0=1013.25):
    """
    Convert meters to hectopascals
    """
    hPa = np.power(1.0-meters*0.0065/288.15, 5.255)*P0
    return hPa

def hPa_to_meters(hPa, P0=1013.25):
    """
    Convert hectopascals to meters
    """
    meters = (288.15 / 0.0065) * (1 - np.power(hPa / P0, 1/5.255))
    return meters

def hybrid_to_hPa(Ps, hya, hyb):
    """
    Convert hybrid level to hectopascals
    Uses hPa = Ps*hyb[i] + hya[i]
    Ps: Ground pressure
    hya: hybrid parameter a in hectopascals (typically around 0-100)
    nyb: hybrid parameter b (unitless)

    """
    hPa = Ps * hyb + hya
    return hPa

def hybrid_to_meters(Ps, hya, hyb, P0=1013.25):
    """
    Convert hybrid level to meters
    First converts to hectopascals, then to meters
    """
    meters = hPa_to_meters(hybrid_to_hPa(Ps, hya, hyb), P0=P0)
    return meters

def gen_hybrid_layers(n_layers, n_sigma_layers, layer_height, P0=1013.25, sigma_exponent=2):
    """
    Generates hybrid layer parameters hya and hyb
    """
    n_pressure_layers=n_layers-n_sigma_layers

    #Sigma layers
    hybi_sigma = np.linspace(1, 0, n_sigma_layers+1)**sigma_exponent
    hyai_sigma = (meters_to_hPa(np.linspace(0, n_sigma_layers*layer_height, n_sigma_layers+1), P0=P0) - P0*hybi_sigma) * 100

    #Pressure layers
    hybi_pres = np.zeros(n_pressure_layers)
    hyai_pres = meters_to_hPa(np.linspace((n_sigma_layers+1)*layer_height, n_layers*layer_height, n_pressure_layers), P0=P0) * 100

    #Concatenate sigma and pressure layers
    hybi = np.concatenate((hybi_sigma, hybi_pres))
    hyai = np.concatenate((hyai_sigma, hyai_pres))

    return hyai, hybi

def plot_hybrid_levels_meter(P0, hyai, hybi, volcano_height=5000):
    """
    Plots hybrid levels in meters
    """
    x = np.linspace(0, 100, 1000)
    y = volcano_height*np.exp(-0.5*(((x/100-0.5)/0.1)**2))
    #Calculate ground pressure
    Ps = meters_to_hPa(y, P0=P0)
    for i in range(len(hyai)):
        plt.plot(x, hybrid_to_meters(Ps, hyai[i], hybi[i]), '-', label="Layer {:d}".format(i))

def plot_hybrid_levels_hPa(P0, hyai, hybi, volcano_height=5000):
    """
    Plots hybrid levels in hectopascals
    """
    x = np.linspace(0, 100, 1000)
    y = volcano_height*np.exp(-0.5*(((x/100-0.5)/0.1)**2))
    #Calculate ground pressure
    Ps = meters_to_hPa(y, P0=P0)
    for i in range(len(hyai)):
        plt.plot(x, hybrid_to_hPa(Ps, hyai[i], hybi[i]), '-', label="Layer {:d}".format(i))

def degree_to_meters_WGS84(lat_degrees):
    """
    Calculates length of a degree in meters. Adapted from Wikipedia.
    """
    #Accurate to 0.01 meter
    phi = lat_degrees / 180 * np.pi
    lat_length = 111132.92 - 559.82*np.cos(2*phi) + 1.175*np.cos(4*phi) - 0.0023*np.cos(6*phi)
    lon_length = 111412.84*np.cos(phi) - 93.5*np.cos(3*phi) + 0.118*np.cos(5*phi)

    return lon_length, lat_length
    
def great_circle_distance_in_meters(lon1, lat1, lon2, lat2):
    """
    Calculates distance from (lon1, lat1) to (lon2, lat2)
    Adapted from wikipedia
    """
    #Convert from lat lon decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d_sigma = 2* np.arcsin ( np.sqrt(np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2 ))

    earth_radius_in_meters=6371137

    return earth_radius_in_meters*d_sigma

def get_grid_cell_size(latitude, longitude):
    """
    Calculates size of each grid cell
    """
    lon_m, lat_m = degree_to_meters_WGS84(latitude)
    lon_res = np.mean(np.diff(longitude))
    lat_res = np.mean(np.diff(latitude))
    assert (lon_res == longitude[1]-longitude[0])
    assert (lat_res == latitude[1]-latitude[0])
    grid_cell_size = (lon_m * lon_res)*(lat_m * lat_res)

    return np.matlib.repmat(grid_cell_size, longitude.size, 1).T

def get_hybrid_layer_thickness(Ps, layer, hyai, hybi, P0=1013.25):
    """
    Computes the thickness of a layer using the bottom pressure P0.
    """
    meters_0 = hybrid_to_meters(Ps, hyai[layer], hybi[layer], P0=P0)
    meters_1 = hybrid_to_meters(Ps, hyai[layer+1], hybi[layer+1], P0=P0)

    return meters_0 - meters_1


def delete_in_place(matrix, to_keep_rows=None, to_keep_cols=None, selftest=True):
    """Reshapes in place by moving data (slow but requires far less memory)."""
    rows, cols = matrix.shape

    #Perform a self test by default to check that the approach still works as intended
    if (selftest):
        a = np.random.randint(low=0, high=5, size=(50, 30))
        st_rows = np.round(np.random.random((a.shape[0])).ravel()).astype(np.bool)
        st_cols = np.round(np.random.random((a.shape[1])).ravel()).astype(np.bool)
        c = a[st_rows,:]
        c = c[:,st_cols]
        b = delete_in_place(a, st_rows, st_cols, selftest=False)
        b.resize((np.count_nonzero(st_rows), np.count_nonzero(st_cols)))
        assert np.array_equal(b, c), "Something is wrong with delete_in_place"
        assert b.__array_interface__['data'][0] == a.__array_interface__['data'][0]

    #Remove rows in place
    if to_keep_rows is not None:
        k=0
        for l, keep in enumerate(to_keep_rows):
            if (keep):
                matrix[k,:] = matrix[l,:]
                k += 1
            else:
                pass
        rows = k

    #Remove cols in place
    if to_keep_cols is not None:
        src = np.flatnonzero(to_keep_cols)
        nnz = len(src)
        dst = np.arange(nnz)

        flat = matrix.ravel()
        for k in range(rows):
            s = src+k*cols
            d = dst+k*nnz
            flat[d] = flat[s]
        cols = nnz

    matrix.resize((rows, cols), refcheck=False)
    return matrix
    #return np.resize(matrix, (rows, cols))



def resample_1D(data, old_boundaries, new_boundaries):
    """Resamples from old bins to new bins"""
    #0-pad data
    eps = 10
    data = np.hstack((0, data, 0))
    old_boundaries = np.hstack((old_boundaries[0]-eps, old_boundaries, old_boundaries[-1]+eps))

    #Create 0-order interpolating polynomial (that extrapolates 0 outside defined region)
    poly = PPoly(data.reshape(1,-1), old_boundaries)

    #Integrate polynomial using new boundaries
    new_data = np.zeros(new_boundaries.shape[0]-1)
    for i in range(len(new_data)):
        new_data[i], _ = quad(poly, new_boundaries[i], new_boundaries[i+1]) / (new_boundaries[i+1] - new_boundaries[i])

    return new_data

def resample_2D(data, old_boundaries, new_boundaries):
    """Resamples from old bins to new bins"""
    new_data = np.zeros((data.shape[0], new_boundaries.shape[0]-1))
    for i in range(data.shape[0]):
        new_data[i,:] = resample_1D(data[i,:], old_boundaries, new_boundaries)

    return new_data
