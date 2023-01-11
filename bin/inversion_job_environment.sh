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

####################
# Options for qsub #
####################

## Name of the job
#$ -N "@TAG@-ashinv"

## Resource use
#$ -l h_rt=@RUNTIME@
#$ -l h_vmem=@MEMORY@
#$ -pe @PARALLEL_ENVIRONMENT@


## Queue to submit in (see qstat -g c for possibilities)
#$ -q @QUEUE@

## Mail to
#$ -M @USER@@met.no

## Project name
#$ -P @PROJECT@

## Shell
#$ -S /bin/bash

## Mail when aborted / rescheduled
#$ -m a

## Reservation yes
#$ -R y

## Standard out logfile
#$ -o "@RUN_DIR@/cout.log"

## Standard error logfile
#$ -e "@RUN_DIR@/cerr.log"

#Check that this script has been set  up properly
if [[ ! -d "@SCRIPT_DIR@" ]]; then
    echo "ERROR: This script is not intended to be called."
    echo "ERROR: Plase use inversion_job_setup.sh to set up an inversion job"
    exit -1
fi

######################################
# Stop on first error unless sourced #
######################################
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e 
    export ASHINV_CLEANUP=1
fi

######################
# Environment to use #
######################
export SCRIPT_DIR="@SCRIPT_DIR@"
export RUN_DIR="@RUN_DIR@"
export TAG="@TAG@"
export RANDOM_SEED="@RANDOM_SEED@"

export RUN_PLUME_HEIGHTS="@RUN_DIR@/plume_heights_@TAG@.csv"
export RUN_CONF_A_PRIORI="@RUN_DIR@/conf_a_priori_@TAG@.ini"
export RUN_CONF_MATCH_FILES="@RUN_DIR@/conf_match_files_@TAG@.ini"
export RUN_CONF_INVERSION="@RUN_DIR@/conf_inversion_@TAG@.ini"
export RUN_OBSERVATIONS="@RUN_DIR@/ash_observations_@TAG@.csv"
export RUN_SIMULATIONS="@RUN_DIR@/ash_simulations_@TAG@.csv"

export INVERSION_ENVIRONMENT_SETUP=1

#################
# Set up Python #
#################
source $SCRIPT_DIR/inversion_job_conda.sh


#########################################
# Call actual job script if not sourced #
#########################################
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    $SCRIPT_DIR/inversion_job_script.sh $*
fi
