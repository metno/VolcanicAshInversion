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

# This script is used to execute the actual inversion job.
# The relevant environment variables need to be set first.

set -e #Stop on first error
set -o pipefail # Enable fail error codes on pipe to tee

#Double check that things appear to be sensible before continuing
if [[ -z "$INVERSION_ENVIRONMENT_SETUP" ]]; then
    echo "ERROR: Environment has not been set up correctly"
    echo "This script is not intended to be run directly"
    echo "Please set up environment properly first."
    exit -1
fi

RUN_DATE=$(date +"%Y%m%dT%H%MZ")
SYSTEM_MATRIX_FILE="$OUT_DIR/inversion_system_matrix.npz"
RESULTS_DIR="$OUT_DIR/results/${TAG}_${RUN_DATE}"
LOGFILE="$RESULTS_DIR/log_running.txt"

#Make output directory and
#copy config there (for reproducibility)
mkdir -p "$RESULTS_DIR"
cp "$RUN_PLUME_HEIGHTS"    "$RESULTS_DIR"
cp "$RUN_CONF_A_PRIORI"    "$RESULTS_DIR"
cp "$RUN_CONF_MATCH_FILES" "$RESULTS_DIR"
cp "$RUN_CONF_INVERSION"   "$RESULTS_DIR"
cp "$RUN_OBSERVATIONS"     "$RESULTS_DIR"
cp "$RUN_SIMULATIONS"      "$RESULTS_DIR"

#Function that echoes to screen & log before executing
function inv_exec {
    echo "==== $(date +"%Y%m%dT%H%MZ"): '$@'"
    echo "==== $(date +"%Y%m%dT%H%MZ"): '$@'" >> "$LOGFILE"
    $@ 2>&1 | tee --append "$LOGFILE"
    if [ $? != 0 ]; then
        exit $?
    fi
}

#Clean up on exit/error
function cleanup_job {
    if [ ! $? -eq 0 ]; then
        echo "ERROR: Something failed!"
    fi

    if [ -e $LOGFILE ]; then
        NEW_LOGFILENAME=$(echo $LOGFILE | sed 's/running/failed/')
        mv $LOGFILE $NEW_LOGFILENAME
    fi
}
trap cleanup_job ERR EXIT KILL SIGTERM


#Set python path
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"

echo ""
echo ""
echo ""
inv_exec echo "==== Inversion run begin ===="


# Create a priori information from plume heights estimate
inv_exec $SCRIPT_DIR/AshInv/APriori.py \
                    --config "$RUN_CONF_A_PRIORI" \
                    --plume_heights_file "$RUN_PLUME_HEIGHTS" \
                    --a_priori_file "$RESULTS_DIR/a_priori.json"
inv_exec echo "INFO: Done creating a priori values"



#Match observations with simulation files
inv_exec $SCRIPT_DIR/AshInv/MatchFiles.py \
                    --config "$RUN_CONF_MATCH_FILES" \
                    --simulation_csv "$RUN_SIMULATIONS" \
                    --observation_csv "$RUN_OBSERVATIONS" \
                    --output_dir "$OUT_DIR/matched_files"
inv_exec echo "INFO: Done matching files"






#Run inversion procedure
#SOLVERS=("direct" "inverse" "pseudo_inverse" "lstsq" "lstsq2" "nnls" "nnls2" "lsq_linear" "lsq_linear2")
SOLVERS=("direct" "inverse" "pseudo_inverse" "lstsq2" "nnls" "nnls2" "lsq_linear2")
for SOLVER in "${SOLVERS[@]}"; do

    SYSTEM_MATRIX=""
    if [ -e $SYSTEM_MATRIX_FILE ]; then
        inv_exec echo "*** WARNING ***: Using existing system matrix"
        SYSTEM_MATRIX="--input_unscaled=$SYSTEM_MATRIX_FILE"
    fi

    inv_exec echo "INFO: Using solver $SOLVER"
    inv_exec $SCRIPT_DIR/AshInv/AshInversion.py \
                    --config "$RUN_CONF_INVERSION" \
                    --matched_files "$OUT_DIR/matched_files/matched_files.csv" \
                    --a_priori_file "$RESULTS_DIR/a_priori.json" \
                    --solver $SOLVER \
                    $SYSTEM_MATRIX \
                    --output_dir "$RESULTS_DIR/$SOLVER"

    inv_exec echo "INFO: $SOLVER done, creating a posteriori emission source file"
    inv_exec $SCRIPT_DIR/AshInv/APosteriori.py \
                    --variable 'a_priori_2d' \
                    --output "$RESULTS_DIR/$SOLVER/a_priori.csv" \
                    --json "$RESULTS_DIR/$SOLVER/inversion_a_posteriori.json"
    inv_exec $SCRIPT_DIR/AshInv/APosteriori.py \
                    --variable 'a_posteriori_2d' \
                    --output "$RESULTS_DIR/$SOLVER/a_posteriori.csv" \
                    --json "$RESULTS_DIR/$SOLVER/inversion_a_posteriori.json"
    inv_exec $SCRIPT_DIR/AshInv/Plot.py \
                    --plotsum=False \
                    --colormap birthe \
                    --usetex=False \
                    --json "$RESULTS_DIR/$SOLVER/inversion_a_posteriori.json" \
                    --output "$RESULTS_DIR/$SOLVER/inversion_a_posteriori.png"

    #Symlink system matrix for subsequent runs.
    if [ ! -e $SYSTEM_MATRIX_FILE ]; then
        inv_exec ln -s "$RESULTS_DIR/$SOLVER/inversion_system_matrix.npz" $SYSTEM_MATRIX_FILE
    fi
done

inv_exec echo "INFO: Done with inversion procedure"






#Move logfile to completed
inv_exec echo "==== Inversion run end ===="
NEW_LOGFILENAME=$(echo $LOGFILE | sed 's/running/completed/')
mv $LOGFILE $NEW_LOGFILENAME
