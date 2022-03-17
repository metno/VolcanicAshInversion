#!/bin/bash

##############################################################################
#                                                                            #
#    This file is part of PVAI - Python Volcanic Ash Inversion.              #
#                                                                            #
#    Copyright 2021, 2022, André R. Brodtkorb <andre.brodtkorb@oslomet.no>   #
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

#set -x #echo commands (useful for debugging)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script is used to set up the conda environment and is called"
    echo "automatically by the inversion procedure."
    echo ""
    echo "ERROR: This script is not intended to be executed."
    echo "INFO: Execute 'source ${BASH_SOURCE[0]}' instead (if you know what you are doing)"
    exit -1
fi

if [ "$CONDA_INITIALIZED" == "1" ]; then
    echo "Conda already initialized"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    CONDA_EXE=${CONDA_EXE:-$SCRIPT_DIR/miniconda3/bin/conda}
    CONDA_ENV_NAME="ash_inversion_env"

    # Check that conda has been set up correctly
    if [[ ! -f $CONDA_EXE ]]; then
        echo "Could not find $CONDA_EXE, searching for conda in path"
        if [[ $(type -P "conda") ]]; then
            CONDA_EXE="$(type -P "conda")"
            echo "Using conda in path: $CONDA_EXE"
        else
            echo "ERROR: Conda has not been set up correctly! Please download and install conda"
            exit -1
        fi
    fi

    #Initialize conda
    eval "$($CONDA_EXE shell.bash hook)"
    #source "$(dirname $CONDA_EXE)/../etc/profile.d/conda.sh"

    if [[ $(conda activate $CONDA_ENV_NAME && echo "SUCCESS") != SUCCESS ]]; then
        echo "ERROR: Conda environment has not been set up correctly! Setting up automatically"
        conda create --name "$CONDA_ENV_NAME" --file "$SCRIPT_DIR/ash_inv.yml"
    fi

    # Setup conda environment
    conda activate $CONDA_ENV_NAME
    
    #Add ash inversion scripts to path
    export ASHINV_OLD_PATH=$PATH
    export PATH=$PATH:$SCRIPT_DIR/AshInv

    #Clean up function
    function cleanup_conda {
        if [ $CONDA_INITIALIZED == 1 ]; then
            if [ ! $? -eq 0 ]; then
                echo "ERROR: Something failed!"
            fi
            echo "INFO: Deactivating Conda"
            conda deactivate
            export CONDA_INITIALIZED=0
            echo "INFO: Resetting path"
            export PATH=$ASHINV_OLD_PATH
        fi
    }

    #Cleanup conda on exit
    if [ "$ASHINV_CLEANUP" == 1 ]; then
        trap cleanup_conda ERR EXIT KILL SIGTERM
    else
        echo "INFO: use cleanup_conda to reset shell"
    fi
    
    #Add ash inversion package to pythonpath
    export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}$(realpath "$SCRIPT_DIR/..")

    export CONDA_INITIALIZED=1
fi
