#!/bin/bash

echo "Run this on Nebula"


set -x

module use /home/met-software/modulefiles
module load met-modules fimex

echo fimex --input.file ../eyja-20200303_first/20100414T0000Z/ASHinv20100414T0000Z_hour.nc --input.printNcML=hyab.ncml

echo "Then manually edit the file hyab.ncml"

echo fimex \
    --input.file=/home/sm_andbr/data/ash_inversion/meteoMACC14_historic/2010_00UTC/meteo20100401_00.nc \
    --input.config=hyab.ncml \
    --output.file=/home/sm_andbr/data/ash_inversion/meteoMACC14_historic_hyb/meteo20100401_00.nc \
    --output.type=nc4

OUT_DIR="/home/sm_andbr/data/ash_inversion/meteoMACC14_historic_hyb/2010_00UTC"
IN_DIR="/home/sm_andbr/data/ash_inversion/meteoMACC14_historic/2010_00UTC"
mkdir -p $OUT_DIR
for abspath in $(ls $IN_DIR/*.nc); do
    filename=$(basename $abspath);
    path=$(dirname $abspath);
    fimex --input.file=$abspath --input.config=hyab.ncml --output.file=$OUT_DIR/$filename --output.type=nc4
done


OUT_DIR="/nobackup/forsk/sm_andbr/AshInversion/Eyja/MACC14.ASH-2010eyja/meteorology/"
TMP_DIR="/home/sm_andbr/nobackup/AshInversion/Eyja/MACC14.ASH-2010eyja/meteorology/tmp"
IN_DIR="/home/sm_andbr/data/ash_inversion/meteoMACC14_historic/2010_00UTC"
mkdir -p $OUT_DIR
mkdir -p $TMP_DIR
for abspath in $(ls $IN_DIR/*.nc); do
    filename=$(basename $abspath);
    path=$(dirname $abspath);
    fimex --input.file=$abspath --input.config=Ps.ncml \
        --extract.selectVariables=PS \
        --extract.selectVariables=map_factor_i \
        --extract.selectVariables=map_factor_j \
        --output.file $TMP_DIR/$filename \
        --output.type=nc4
done

module load NCO/4.7.9-nsc5
ncrcat -h -L 1 $TMP_DIR/*.nc $OUT_DIR/pressure.nc
