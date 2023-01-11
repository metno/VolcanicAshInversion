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

set -e #Stop on first error
#set -x #echo commands (useful for debugging)

#Absolute path of script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


#Read internal config that sets some of the default parameters
DEFAULTS_FILE="$(realpath $SCRIPT_DIR/..)/internal/inversion_defaults.sh"
if [[ -f "$DEFAULTS_FILE" ]]; then
    echo "Using defaults from $DEFAULTS_FILE"
    source $DEFAULTS_FILE
else
    echo "Warning: Defaults file $DEFAULTS_FILE does not exist. "
fi


#Initialize defaults
CONF_DIR=${CONF_DIR:-NONE}
RUN_DIR=${RUN_DIR:-NONE}
TAG=${TAG:-NONE}
QUEUE=${QUEUE:-NONE}       #qsub queue to submit to
PROJECT=${PROJECT:-NONE}   #qsub project to use
RUNTIME=${RUNTIME:-1:30:0} #qsub max job run time
MEMORY=${MEMORY:-1G}       #qsub memory to use
PARALLEL_ENVIRONMENT=${PARALLEL_ENVIRONMENT:-NONE} #qsub PE to use
SUBMIT=${SUBMIT:-0}
RANDOM_SEED=$(date +"%s")


function usage {
    echo "This program sets up the ash inversion procedure and submits a job."
    echo "Typical runtime depends on the complexity of the inversion job"
    echo ""
    echo "Usage: $0 [options]"
    echo "Options: "
    echo "     --tag <some_tag>            # Tag your run (use a descriptive tag you remember!)"
    echo "     --conf <dir>                # Directory containing all relevant config files:"
    echo "                                 # "
    echo "     [--run-dir <output_dir>]    # Explicitly set run/output directory"
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
        --run-dir)  [ $# -gt 0 ] && RUN_DIR="$1"  && shift ;;
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
if [ "$RUN_DIR" == "NONE" ]; then
    RUN_DIR=$(realpath $SCRIPT_DIR/..)/output/${TAG}_$(date +"%Y%m%dT%H%MZ")
    echo "INFO: Using output directory '$RUN_DIR'"
    if [ -d $RUN_DIR ]; then
        echo "ERROR: Output directory exists!"
        echo "ERROR: Aborting"
        exit -1
    fi
fi




#Remove trailing slash from conf dir / out dir
CONF_DIR=${CONF_DIR%%/}
RUN_DIR=${RUN_DIR%%/}






echo "INFO: Creating output directory '$RUN_DIR' and copying config files"
mkdir -p $RUN_DIR
for SRC_FILE in $CONF_DIR/*.* $SCRIPT_DIR/inversion_job_environment.sh; do
    FILENAME=$(basename $SRC_FILE)
    DST_FILE="$RUN_DIR/${FILENAME%.*}_${TAG}.${FILENAME##*.}"
    echo "INFO: Copying '$SRC_FILE'"
    cp -L "$SRC_FILE" "$DST_FILE";

    # Substitute all @VAR@ with proper contents of $VAR
    # Check if text file
    if ( grep -qI . $SRC_FILE ); then
        for SUBSTITUTION_VAR in $(cat $DST_FILE | grep -oP "@.*?@" | tr -d '@' | sort | uniq ); do
            echo "INFO: Subsitituting $SUBSTITUTION_VAR => ${!SUBSTITUTION_VAR}"
            sed -i'' "s#@${SUBSTITUTION_VAR}@#${!SUBSTITUTION_VAR}#g" "$DST_FILE"
        done
    fi
done



echo "INFO: Renaming job script"
JOB_SCRIPT="$RUN_DIR/job_script_${TAG}.sh"
mv "$RUN_DIR/inversion_job_environment_${TAG}.sh" $JOB_SCRIPT




echo " "
echo "################ SUMMARY #################"
echo " "
echo "Job has been set up in $RUN_DIR"
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
