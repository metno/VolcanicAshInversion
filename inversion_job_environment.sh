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

####################
# Options for qsub #
####################

## Name of the job
#$ -N "ashinv-@TAG@"

## Resource use
#$ -l h_rt=01:30:00
#$ -l h_vmem=@MEMORY@

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
#$ -o "@OUT_DIR@/cout.log"

## Standard error logfile
#$ -e "@OUT_DIR@/cerr.log"

set -e #Stop on first error

######################
# Environment to use #
######################
export SCRIPT_DIR="@SCRIPT_DIR@"
export OUT_DIR="@OUT_DIR@"
export TAG="@TAG@"

export RUN_PLUME_HEIGHTS="@OUT_DIR@/plume_heights_@TAG@.csv"
export RUN_CONF_A_PRIORI="@OUT_DIR@/conf_a_priori_@TAG@.ini"
export RUN_CONF_MATCH_FILES="@OUT_DIR@/conf_match_files_@TAG@.ini"
export RUN_CONF_INVERSION="@OUT_DIR@/conf_inversion_@TAG@.ini"
export RUN_OBSERVATIONS="@OUT_DIR@/ash_observations_@TAG@.csv"
export RUN_SIMULATIONS="@OUT_DIR@/ash_simulations_@TAG@.csv"

export INVERSION_ENVIRONMENT_SETUP=1

#################
# Set up Python #
#################
source $SCRIPT_DIR/inversion_job_conda.sh
trap cleanup_conda ERR EXIT KILL SIGTERM

##########################
# Call actual job script #
##########################
$SCRIPT_DIR/inversion_job_script.sh
