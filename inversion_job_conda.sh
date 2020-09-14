#!/bin/bash

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

if [[ $_ == $0 ]]; then
    echo "ERROR: Script is a subshell."
    echo "INFO: Execute 'source ${BASH_SOURCE[0]}' instead"
    exit -1
fi

if [ $CONDA_INITIALIZED == 1 ]; then
    echo "Conda already initialized"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    CONDA_EXE="$SCRIPT_DIR/miniconda3/bin/conda"
    CONDA_ENV_PATH="$SCRIPT_DIR/conda_ash_inv"


    # Check that conda has been set up correctly
    if [[ ! -f $CONDA_EXE ]]; then
        echo "ERROR: Conda has not been set up correctly! Please download and install conda"
        exit -1
    fi
    if [[ ! -d $CONDA_ENV_PATH ]]; then
        echo "ERROR: Conda environment has not been set up correctly! Please set up environment"
        exit -1
    fi


    # Setup conda environment
    eval "$($CONDA_EXE shell.bash hook)"
    conda activate $CONDA_ENV_PATH


    #Clean up function
    function cleanup_conda {
        if [ $CONDA_INITIALIZED == 1 ]; then
            if [ ! $? -eq 0 ]; then
                echo "ERROR: Something failed!"
            fi
            echo "INFO: Deactivating Conda"
            conda deactivate
            export CONDA_INITIALIZED=0
        fi
    }

    #Cleanup conda on exit
    trap cleanup_conda ERR EXIT KILL SIGTERM

    export CONDA_INITIALIZED=1
fi
