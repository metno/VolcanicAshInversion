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

# This script is used to execute the actual inversion job.
# The relevant environment variables need to be set first.

set -e #Stop on first error
set -o pipefail # Enable fail error codes on pipe to tee

#Double check that things appear to be sensible before continuing
if [[ -z "$INVERSION_ENVIRONMENT_SETUP" ]]; then
    echo "ERROR: This script is not intended to be called."
    echo "ERROR: Plase use inversion_job_setup.sh to set up an inversion job"
    exit -1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# need to run from script-dir to allow for git
cd $SCRIPT_DIR



#Initialize default arguments
RUN_SETUP=${RUN_SETUP:-0}
RUN_A_PRIORI=${RUN_A_PRIORI:-0}
RUN_MATCH_FILES=${RUN_MATCH_FILES:-0}
RUN_INVERSION=${RUN_INVERSION:-0}
RUN_PLOTS=${RUN_PLOTS:-0}
RUN_A_POSTERIORI_CSV=${RUN_A_POSTERIORI_CSV:-0}
SOLVER=${SOLVER:-"direct"}
RUN_DATE=${RUN_DATE:-$(date +"%Y%m%dT%H%MZ")}
RESULTS_DIR=${RESULTS_DIR:-"$RUN_DIR/results/${TAG}_${RUN_DATE}"}
SYSTEM_MATRIX_FILE=${SYSTEM_MATRIX_FILE:-"$RUN_DIR/inversion_system_matrix.npz"}

function usage {
    echo "This program runs the inversion procedure."
    echo ""
    echo "Usage: $0 [options]"
    echo "Options to override defaults"
    echo "     --results-dir <dir>         # $RESULTS_DIR"
    echo ""
    echo "Options to run only part of system. If none of these are selected, all parts will be run: "
    echo "     --run-setup                 # Run setup (copy files, store source code version, etc)"
    echo "     --run-apriori               # Run a priori generation"
    echo "     --run-matchfiles            # Run match files script (colocate observation and simulation)"
    echo "     --run-inversion             # Run inversion code"
    echo "     --run-plots                 # Run plots"
    echo "     --run-aposteriori-csv       # Create a posteriori csv files (suitable for eEMEP simulations)"
    echo " "
    echo "Options for inversion:"
    echo "     --solver {direct|inverse|pseudo_inverse"
    echo "               |lstsq|lstsq2|nnls|nnls2"
    echo "               |lsq_linear|lsq_linear2} # Select solver to use in inversion"
}

#Get command line options
while [[ $# -gt 0 ]]; do
    key="$1"
    shift
    case $key in
        --run-setup)           RUN_SETUP=1               ;;
        --run-apriori)         RUN_A_PRIORI=1            ;;
        --run-matchfiles)      RUN_MATCH_FILES=1         ;;
        --run-inversion)       RUN_INVERSION=1           ;;
        --run-plots)           RUN_PLOTS=1               ;;
        --run-aposteriori-csv) RUN_A_POSTERIORI_CSV=1    ;;
        --solver)        [ $# -gt 0 ] &&      SOLVER="$1" && shift ;;
        --results-dir)   [ $# -gt 0 ] && RESULTS_DIR=$(realpath "$1") && shift ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$key'"
            usage
            exit -1
            ;;
    esac
done

#Check command line options
if [[ $RUN_SETUP == 0             \
        && $RUN_A_PRIORI == 0    \
        && $RUN_MATCH_FILES == 0 \
        && $RUN_INVERSION == 0   \
        && $RUN_PLOTS == 0 \
        && $RUN_A_POSTERIORI_CSV == 0 ]]; then
    echo "No options selected: running all parts of inversion"
    RUN_SETUP=1
    RUN_A_PRIORI=1
    RUN_MATCH_FILES=1
    RUN_INVERSION=1
    RUN_PLOTS=1
    RUN_A_POSTERIORI_CSV=1
fi





#Set up logging
LOGFILE=${LOGFILE:-"$RESULTS_DIR/log_running.txt"}
function inv_exec {
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir -p "$RESULTS_DIR"
    fi
    echo "==== $(date +"%Y%m%dT%H%MZ"): '$@'"
    echo "==== $(date +"%Y%m%dT%H%MZ"): '$@'" >> "$LOGFILE"
    $@ 2>&1 | tee --append "$LOGFILE"
    if [ $? != 0 ]; then
        exit $?
    fi
}
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

#Set python path. If PYTHONPATH is empty do not add leading :
ASHINV_PYTHONPATH=$(realpath "$SCRIPT_DIR/..")




echo ""
echo ""
echo ""
inv_exec echo "==== Inversion run begin ===="

#Make output directory and
#copy config there (for reproducibility)
if [ $RUN_SETUP == 1 ]; then
    inv_exec echo "INFO: Making results directory $RESULTS_DIR and setting up config files"
    inv_exec mkdir -p "$RESULTS_DIR"
    inv_exec cp "$RUN_PLUME_HEIGHTS"    "$RESULTS_DIR"
    inv_exec cp "$RUN_CONF_A_PRIORI"    "$RESULTS_DIR"
    inv_exec cp "$RUN_CONF_MATCH_FILES" "$RESULTS_DIR"
    inv_exec cp "$RUN_CONF_INVERSION"   "$RESULTS_DIR"
    inv_exec cp "$RUN_OBSERVATIONS"     "$RESULTS_DIR"
    inv_exec cp "$RUN_SIMULATIONS"      "$RESULTS_DIR"

    #Print git version info for reproducibility
    inv_exec echo "INFO: Running git to store source code version and local changes"
    inv_exec echo $(pwd)
    inv_exec git log -n1
    inv_exec git status --porcelain
    inv_exec echo "INFO: Storing uncommited git changes in patch file '$RESULTS_DIR/uncommitted_changes.patch'"
    git diff > "$RESULTS_DIR/uncommitted_changes.patch"  
fi

if [ $RUN_A_PRIORI == 1 ]; then
    # Create a priori information from plume heights estimate
    inv_exec $ASHINV_PYTHONPATH/AshInv/APriori.py \
                        --config "$RUN_CONF_A_PRIORI" \
                        --a_priori_file "$RESULTS_DIR/a_priori.json"
    inv_exec echo "INFO: Done creating a priori values"
fi


if [ $RUN_MATCH_FILES == 1 ]; then
    #Match observations with simulation files
    inv_exec $ASHINV_PYTHONPATH/AshInv/MatchFiles.py \
                        --config "$RUN_CONF_MATCH_FILES" \
                        --simulation_csv "$RUN_SIMULATIONS" \
                        --observation_csv "$RUN_OBSERVATIONS" \
                        --output_dir "$RUN_DIR/matched_files"
    inv_exec echo "INFO: Done matching files"
fi


if [ $RUN_INVERSION == 1 ]; then
    #Run inversion procedure
    inv_exec echo "INFO: Using solver $SOLVER"

    #Check if we have existing system matrix
    SYSTEM_MATRIX=""
    if [ -e $SYSTEM_MATRIX_FILE ]; then
        inv_exec echo "*** WARNING ***: Using existing system matrix"
        SYSTEM_MATRIX="--input_unscaled=$SYSTEM_MATRIX_FILE"
    fi

    #Run inversion (with progress file)
    inv_exec $ASHINV_PYTHONPATH/AshInv/AshInversion.py \
                    --config "$RUN_CONF_INVERSION" \
                    --progress_file "$RUN_DIR/progress.npz" \
                    --matched_files "$RUN_DIR/matched_files/matched_files.csv" \
                    --a_priori_file "$RESULTS_DIR/a_priori.json" \
                    --solver $SOLVER \
                    $SYSTEM_MATRIX \
                    --output_dir "$RESULTS_DIR"

    #Remove progress file
    inv_exec rm -f "$RUN_DIR/progress.npz"

    #Symlink system matrix for subsequent runs.
    if [ ! -e $SYSTEM_MATRIX_FILE ]; then
        inv_exec ln -s "$RESULTS_DIR/inversion_system_matrix.npz" $SYSTEM_MATRIX_FILE
    fi

    inv_exec echo "INFO: solver $SOLVER done"
fi


if [ $RUN_A_POSTERIORI_CSV == 1 ]; then                    
    for RESULT_JSON in $RESULTS_DIR/inversion_*_a_posteriori.json; do
        # Convert a priori once
        if [ ! -f "$RESULTS_DIR/a_priori.csv" ]; then
            inv_exec $ASHINV_PYTHONPATH/AshInv/APosteriori.py \
                            --variable 'a_priori_2d' \
                            --output "$RESULTS_DIR/a_priori.csv" \
                            --json $RESULT_JSON
        fi
    
        # Convert a posteriori to csv
        RESULT_CSV="${RESULT_JSON%.*}".csv
        inv_exec $ASHINV_PYTHONPATH/AshInv/APosteriori.py \
                        --variable 'a_posteriori_2d' \
                        --json $RESULT_JSON \
                        --output $RESULT_CSV
    done
fi



if [ $RUN_PLOTS == 1 ]; then
    for RESULT_JSON in $RESULTS_DIR/inversion_*_a_posteriori.json; do
        RESULT_PNG="${RESULT_JSON%.*}".png
        inv_exec $ASHINV_PYTHONPATH/AshInv/Plot.py \
                        --plotsum=False \
                        --colormap birthe \
                        --usetex=False \
                        --json $RESULT_JSON \
                        --output $RESULT_PNG
    done
fi






#Move logfile to completed
inv_exec echo "==== Inversion run end ===="
NEW_LOGFILENAME=$(echo $LOGFILE | sed 's/running/completed/')
mv $LOGFILE $NEW_LOGFILENAME
