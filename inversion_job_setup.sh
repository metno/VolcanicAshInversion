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

set -e #Stop on first error

#Absolute path of script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


#Read internal config that sets some of the default parameters
DEFAULTS_FILE="$(realpath "$SCRIPT_DIR/../internal/inversion_defaults.sh")"
if [[ -f $DEFAULTS_FILE ]]; then
    echo "Using defaults from $DEFAULTS_FILE"
    source $DEFAULTS_FILE
else
    echo "Warning: No default values found. Continue at your own risk"
fi


#Initialize defaults
CONF_DIR=${CONF_DIR:-NONE}
OUT_DIR=${OUT_DIR:-NONE}
TAG=${TAG:-NONE}
QUEUE=${QUEUE:-NONE}       #qsub queue to submit to
PROJECT=${PROJECT:-NONE}   #qsub project to use
RUNTIME=${RUNTIME:-1:30:0} #qsub max job run time
MEMORY=${MEMORY:-1G}       #qsub memory to use
PARALLEL_ENVIRONMENT=${PARALLEL_ENVIRONMENT:NONE} #qsub PE to use
SUBMIT=${SUBMIT:-0}


function usage {
    echo "This program sets up the ash inversion procedure and submits a job."
    echo "Typical runtime is ~30 minutes depending on complexity of the inversion job"
    echo ""
    echo "Usage: $0 [options]"
    echo "Options: "
    echo "     --tag <some_tag>            # Tag your run (use a descriptive tag you remember!)"
    echo "     --conf <dir>                # Directory containing all relevant config files:"
    echo "                                 # "
    echo "     [--out-dir <output_dir>]    # Explicitly set output directory"
    echo "     [--memory {1G|4G|16G|...}]  # Reserve x GB memory (default '$MEMORY')"
    echo "     [--memory <queue_name>]     # Submit to this queue (default '$QUEUE')"
    echo "     [--submit]                  # Submit using qsub, otherwise just set up job"
}

#Get command line options
while [[ $# -gt 0 ]]; do
    key="$1"
    shift
    case $key in
        --conf)     [ $# -gt 0 ] && CONF_DIR="$1" && shift ;;
        --tag)      [ $# -gt 0 ] && TAG="$1"      && shift ;;
        --out-dir)  [ $# -gt 0 ] && OUT_DIR="$1"  && shift ;;
        --memory)   [ $# -gt 0 ] && MEMORY="$1"   && shift ;;
        --queue)    [ $# -gt 0 ] && QUEUE="$1"   && shift ;;
        --submit)                   SUBMIT=1               ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$key'"
            exit -1
            ;;
    esac
done


if [ "$TAG" == "NONE" ]; then
    usage
    echo "Missing --tag <RUN TAG>"
    exit -1
fi
if [ "$CONF_DIR" == "NONE" ]; then
    usage
    echo "Missing --conf <DIR>"
    exit -1
fi
if [ "$OUT_DIR" == "NONE" ]; then
    OUT_DIR=$SCRIPT_DIR/output/${TAG}_$(date +"%Y%m%dT%H%MZ")
    echo "INFO: Using output directory '$OUT_DIR'"
    if [ -d $OUT_DIR ]; then
        echo "ERROR: Output directory exists!"
        echo "ERROR: Aborting"
        exit -1
    fi
fi




#Remove trailing slash from conf dir / out dir
CONF_DIR=${CONF_DIR%%/}
OUT_DIR=${OUT_DIR%%/}






echo "INFO: Creating output directory '$OUT_DIR' and copying config files"
mkdir -p $OUT_DIR
for FILENAME in "plume_heights.csv"    \
                "conf_a_priori.ini"    \
                "conf_match_files.ini" \
                "conf_inversion.ini"   \
                "ash_observations.csv" \
                "ash_simulations.csv"  \
                ; do
    SRC_FILE="$CONF_DIR/$FILENAME"
    DST_FILE="$OUT_DIR/${FILENAME%.*}_${TAG}.${FILENAME##*.}"
    echo "INFO: Copying '$SRC_FILE'"
    if [ ! -e "$SRC_FILE" ]; then
        echo "ERROR: Could not find $SRC_FILE"
        exit -1
    else
        cp -L "$SRC_FILE" "$DST_FILE";
    fi
done





echo "INFO: Setting up job script"
# This reads the job script template, and replaces all
# @VAR@ with contents of $VAR
JOB_SCRIPT="$OUT_DIR/job_script_${TAG}.sh"
cp -L "$SCRIPT_DIR/inversion_job_environment.sh" $JOB_SCRIPT
chmod +x "$JOB_SCRIPT"
for SUBSTITUTION_VAR in $(cat $JOB_SCRIPT | grep -oP "@.*?@" | tr -d '@' | sort | uniq ); do
    echo "INFO: Subsitituting $SUBSTITUTION_VAR => ${!SUBSTITUTION_VAR}"
    sed -i'' "s#@${SUBSTITUTION_VAR}@#${!SUBSTITUTION_VAR}#g" "$JOB_SCRIPT"
done




echo " "
echo "################ SUMMARY #################"
echo " "
echo "Job has been set up in $OUT_DIR"
if [ $SUBMIT == 1 ]; then
    echo "Submitting job!"
    qsub "$JOB_SCRIPT"
else
    echo "Please check files in directory and then:"
    echo "* Execute 'qsub $JOB_SCRIPT' to submit job."
fi
echo "* Execute 'qdel <job id>' to delete job"
echo "* Execute 'watch qstat -j <jobid>' to see progress"
echo " "
echo "############## END SUMMARY ###############"

exit 0
