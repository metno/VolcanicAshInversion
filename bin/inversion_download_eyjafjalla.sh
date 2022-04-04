#!/bin/bash

##############################################################################
#                                                                            #
#    This file is part of PVAI - Python Volcanic Ash Inversion.              #
#                                                                            #
#    Copyright 2022, André R. Brodtkorb <andre.brodtkorb@oslomet.no>         #
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

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ASHINV_CLEANUP=1 source "$SCRIPT_DIR/inversion_job_conda.sh"

echo "This script will download data from zenodo and take a long time to complete"
echo "please be patient"

sleep 3

#Download forward runs
mkdir -p "$SCRIPT_DIR/../forward_runs"
cd "$SCRIPT_DIR/../forward_runs"
zenodo_get 3818196 

#Download satellite observations
mkdir -p "$SCRIPT_DIR/../satellite"
cd "$SCRIPT_DIR/../satellite"
zenodo_get 3855526

#Unzip satellite observations
for filename in *.tar.bz2; do
    tar jxvf $filename;
done;
