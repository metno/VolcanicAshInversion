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


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import datetime
import json
import os



def makePlotFromJson(json_filename, outfile, colormap):
    emission_times, level_heights, a_priori, a_posteriori, meta = readJson(json_filename)
    fig = plotAshInv(emission_times, level_heights, a_priori, a_posteriori, colormap=colormap)
    saveFig(outfile, fig, meta)


def readJson(json_filename):
    #Read data
    with open(json_filename, 'r') as infile:
        json_string = infile.read()

    #Parse data
    json_data = json.loads(json_string)

    #Copy only data we care about
    emission_times = np.array(json_data["emission_times"], dtype='datetime64[ns]')
    level_heights = np.array(json_data["level_heights"], dtype=np.float64)

    ordering_index = np.array(json_data["ordering_index"], dtype=np.int64)
    a_priori_2d = np.array(json_data["a_priori_2d"], dtype=np.float64)
    a_posteriori_2d = np.array(json_data["a_posteriori_2d"], dtype=np.float64)

    residual = np.array(json_data["residual"], dtype=np.float64)
    convergence = np.array(json_data["convergence"], dtype=np.float64)

    arguments = json_data["arguments"]
    config = json_data["config"]
    run_date = np.array(json_data["run_date"], dtype='datetime64[ns]')

    #Prune unused a priori data
    any_valid = np.flatnonzero(np.sum(ordering_index >= 0, axis=0))
    ordering_index = ordering_index[:,any_valid.min():any_valid.max()+1]
    emission_times = emission_times[any_valid.min():any_valid.max()+1]

    #Make JSON-data into 2d matrix
    x = expandVariable(emission_times, level_heights, ordering_index, a_posteriori_2d)
    x_a = expandVariable(emission_times, level_heights, ordering_index, a_priori_2d)

    return emission_times, level_heights, x_a, x, json.dumps(json_data, indent=4)



def expandVariable(emission_times, level_heights, ordering_index, variable):
    #Make JSON-data into 2d matrix
    x = np.ma.masked_all(ordering_index.shape)
    for t in range(len(emission_times)):
        for a in range(len(level_heights)):
            emis_index = ordering_index[a, t]
            if (emis_index >= 0):
                x[a, t] = variable[emis_index]
    return x



def saveFig(filename, fig, metadata):
    with PdfPages(filename) as pdf:
        pdf.attach_note(metadata, positionRect=[0, 0, 100, 100])
        pdf.savefig(fig)


def plotAshInv(emission_times, level_heights, a_priori, a_posteriori, fig=None, usetex=False, colormap='default'):

    def npTimeToDatetime(np_time):
        return datetime.datetime.utcfromtimestamp((np_time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

    def npTimeToStr(np_time, fmt="%Y-%m-%d %H:%M"):
        return npTimeToDatetime(np_time).strftime(fmt)

    #Make axis labels
    x_ticks = np.arange(0, emission_times.size)
    x_labels = [npTimeToStr(t, fmt="%d %b\n%H:%M") for t in emission_times]
    x_ticks = x_ticks[3::8]
    x_labels = x_labels[3::8]

    y_ticks = np.arange(0, level_heights.size)
    y_labels = ["{:.0f}".format(a) for a in np.cumsum(level_heights) - level_heights[0]/2]
    y_ticks = y_ticks[::2]
    y_labels = y_labels[::2]

    if (colormap == 'default'):
        colors = [
            (0.0, (1.0, 1.0, 0.8)),
            (0.05, (0.0, 1.0, 0.0)),
            (0.4, (0.9, 1.0, 0.2)),
            (0.6, (1.0, 0.0, 0.0)),
            (1.0, (0.6, 0.2, 1.0))
        ]
    elif (colormap == 'alternative'):
        colors = [
            (0.0, (1.0, 1.0, 0.6)),
            (0.4, (0.9, 1.0, 0.2)),
            (0.6, (1.0, 0.8, 0.0)),
            (0.7, (1.0, 0.4, 0.0)),
            (0.8, (1.0, 0.0, 0.0)),
            (0.9, (1.0, 0.2, 0.6)),
            (1.0, (0.6, 0.2, 1.0))
        ]
    elif (colormap == 'birthe'):
        colors = [
            ( 0/35, ("#ffffff")),
            ( 4/35, ("#b2e5f9")),
            (13/35, ("#538fc9")),
            (18/35, ("#47b54c")),
            (25/35, ("#f5e73c")),
            (35/35, ("#df2b24"))
        ]
    cm = LinearSegmentedColormap.from_list('ash', colors, N=256)
    cm.set_bad(alpha = 0.0)



    y_max = max(1.0e-10, 1.3*max(a_priori.sum(axis=0).max(), a_posteriori.sum(axis=0).max()))

    diff = a_posteriori-a_priori
    diff_range = np.max(np.abs(diff))
    x_range = max(a_priori.max(), a_posteriori.max())

    if (fig is None):
        fig = plt.figure(figsize=(14,4), dpi=200)

    # First subfigure
    ax1 = plt.subplot(1,3,1)
    plt.title("A priori")
    plt.imshow(a_priori, aspect='auto', interpolation='none', origin='lower', cmap=cm, vmin=0.0, vmax=x_range)
    plt.colorbar(orientation='horizontal', pad=0.15)
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0, horizontalalignment='center', usetex=usetex)
    plt.yticks(ticks=y_ticks, labels=y_labels, usetex=usetex)

    ax2 = ax1.twinx()
    ax2.autoscale(False)
    plt.plot(a_priori.sum(axis=0), 'kx--', linewidth=2, alpha=0.5, label='A priori')
    plt.ylim(0, y_max)
    plt.grid()
    plt.legend()

    #Second subfigure
    ax1 = plt.subplot(1,3,2)
    plt.title("Inverted")
    plt.imshow(a_posteriori, aspect='auto', interpolation='none', origin='lower', cmap=cm, vmin=0.0, vmax=x_range)
    plt.colorbar(orientation='horizontal', pad=0.15)
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0, horizontalalignment='center', usetex=usetex)
    plt.yticks(ticks=y_ticks, labels=y_labels, usetex=usetex)

    ax2 = ax1.twinx()
    ax2.autoscale(False)
    plt.plot(a_priori.sum(axis=0), 'kx--', linewidth=2, alpha=0.5, label='A priori')
    plt.plot(a_posteriori.sum(axis=0), 'ko-', fillstyle='none', label='Inverted')
    plt.ylim(0, y_max)
    plt.grid()
    plt.legend()

    #Third subfigure
    plt.subplot(1,3,3)
    plt.title("Inverted - A priori")
    plt.imshow(diff, aspect='auto',
        interpolation='none',
        origin='lower',
        cmap='bwr', vmin=-diff_range, vmax=diff_range)
    plt.colorbar(orientation='horizontal', pad=0.15)
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0, horizontalalignment='center', usetex=usetex)
    plt.yticks(ticks=y_ticks, labels=y_labels, usetex=usetex)
    plt.grid()

    #Set tight layout to minimize overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig




def plotAshInvMatrix(matrix, fig=None, downsample=True):
    def prime_factors(n):
        """ Finds prime factors of n  """
        i = 2
        factors = []
        while i * i <= n:
            if (n % i) != 0:
                i += 1
            else:
                n = n // i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def find_downsample_factor(shape, output):
        """ Finds a downsample factor so that shape is "close" to output """
        factors = prime_factors(int(shape))
        factor = 1
        for f in factors:
            if ((shape / (factor*f)) > output):
                factor *= f
            else:
                break
        return factor

    def rebin(arr, new_shape):
        """Rebin 2D array arr to shape new_shape by averaging."""
        if (new_shape[0] < arr.shape[0] or new_shape[1] < arr.shape[1]):
            shape = (new_shape[0], arr.shape[0] // new_shape[0],
                    new_shape[1], arr.shape[1] // new_shape[1])
            return arr.reshape(shape).mean(-1).mean(1)
        else:
            return arr

    def downsample(arr, target):
        """ Resizes an image to something close to target """
        target_size = [s // find_downsample_factor(s, t) for s, t in zip(arr.shape, target)]
        return rebin(arr, target_size)


    if (fig is None):
        fig = plt.figure(figsize=(18, 18))


    #Downsample M to make (much) faster to plot...
    fig_size = fig.get_size_inches()*fig.dpi
    m = matrix
    if (downsample):
        m = downsample(matrix, fig_size)

    #For plotting, force negative numbers to zero
    m[m<0] = 0.0

    plt.imshow(m,
        aspect='auto',
        interpolation='none',
        norm=LogNorm(vmin=max(matrix.min(), 1e-10), vmax=max(matrix.max(), 2e-10)),
        extent=[0, matrix.shape[1], matrix.shape[0], 0])
    plt.title('Matrix')
    plt.colorbar()
    plt.xlabel('Emission number')
    plt.ylabel('Observation number')

    return fig


if __name__ == "__main__":
    import configargparse

    plt.rc('font',**{'family':'serif','serif':['Times']})
    plt.rc('text', usetex=True)

    parser = configargparse.ArgParser(description='Plot from ash inversion.')
    parser.add("-j", "--json", type=str, help="JSON-file to plot", default=None, required=True)
    parser.add("-o", "--output", type=str, help="Output file", default=None)
    parser.add("-c", "--colormap", type=str, help="Colormap to use", default='default')
    args = parser.parse_args()

    print("Arguments: ")
    print("=======================================")
    for var, val in vars(args).items():
        print("{:s} = {:s}".format(var, str(val)))
    print("=======================================")

    outfile = args.output
    if outfile is None:
        basename, ext = os.path.splitext(os.path.abspath(args.json))
        outfile = basename + ".pdf"

    if (args.json is not None):
        print("Writing output to " + outfile)
        makePlotFromJson(args.json, outfile, args.colormap)
