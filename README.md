# VolcanicAshInversion
This repository contains the volcanic ash inversion source code written in Python. 

The source code is used to take a set of forward runs with unit emissions, and then find the combination of these that matches observations from sattelite images best. 

The code first generates an a priori estimate of ash emissions from observations of ash cloud heihgt. It then continues by colocating ash simulatinons with the observations, and finally assemble these source (forward simulation) - receptor (satellite image) relationships into a large source-receptor matrix. We then use least squares with Tikhonov regularization to find the emissions that match the observatios best. 

# Running example
The following helps you set up a minimal working example of the source code and run a simple inversion. 

This documentation assumes that you have the following directory structure after having followed all the steps:
```
VolcanicAsh
`-> VolcanicAshInversion
|   `-> AshInv
|   |   `-> AshInversion.py
|   |   `-> ...
|   `-> bin
|   |   `-> inversion_job_setup.sh
|   |   `-> ...
|   |   `-> inversion.sh
|   `-> ash_inv.yml
|   `-> ...
`-> forward_runs
|   `-> ASHinv20100414T0000Z_hourInst.nc
|   `-> ...
|   `-> ASHinv20100417T2100Z_hourInst.nc
`-> satellite
    `-> lognormal_1_5
        `-> MSG2-ASH-IcelandEurope_1088_0088_1537x494-201004161900_ExtendedVOLE_lognormal_1_5_gridded.nc
        `-> ...
        `-> MSG2-ASH-IcelandEurope_1088_0088_1537x494-201004161900_ExtendedVOLE_lognormal_1_5_gridded.nc
```

Start by creating the parent directory `VolcanicAsh`
```
mkdir VolcanicAsh
```

## Cloning source code

The source code can be downloaded from github
```
cd VolcanicAsh
git clone https://github.com/babrodtk/VolcanicAshInversion.git
```
should clone the source code from Github. Now test that you get expected results from the regression tests:
```
cd VolcanicAsh/VolcanicAshInversion
bin/inversion_test.sh
```
should give something like the following:
```
=============================================================================== test session starts ================================================================================
platform linux -- Python 3.7.3, pytest-6.2.5, py-1.11.0, pluggy-1.0.0
rootdir: /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/AshInv
plugins: dependency-0.5.1
collected 4 items                                                                                                                                                                  

tests/AshInversion_test.py ....                                                                                                                                              [100%]

================================================================================ 4 passed in 17.90s ================================================================================
INFO: Deactivating Conda
INFO: Resetting path
```

## Downloading data

The data is arcived on Zenodo, and can be downloaded using zenodo_get. Start by installing zenodo_get:
* In python using PiP `pip install zenodo_get`
* or using conda `conda install zenodo_get` 

When you have installed zenodo_get, you can download the data for the forward simulations
```
cd VolcanicAsh
mkdir forward_runs
cd forward_runs

#Download forward runs
zenodo_get 3818196 
```
and for the satellite observations:
```
cd VolcanicAsh
mkdir satellite
cd satellite

#Download satellite observations
zenodo_get 3855526

#Unzip one of te satellite observation datasets
tar jxvf lognormal_1_5.tar.bz2
```

## Setting up an inversion job
The volcanic ash source code can set up a job for you. The scripts copy the configuration into a separate directory, and run the inversion from there for reproducibility. It is very easy to get mixed up when running multiple inversions (e.g., when experimenting with parameters), so the system is set up to be able to track what parameters were used to generate a given result.
```
cd VolcanicAsh/VolcanicAshInversion
bin/inversion_job_setup.sh --tag EYJAFJALLA --conf input_data/eyja/conf_real
```
This should then give something like the following output:
```
Warning: Defaults file /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/internal/inversion_defaults.sh does not exist. 
INFO: Using output directory '/home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z'
INFO: Creating output directory '/home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z' and copying config files
INFO: Copying 'input_data/eyja/conf_real/a_priori.dat'
INFO: Copying 'input_data/eyja/conf_real/ash_observations.csv'
INFO: Copying 'input_data/eyja/conf_real/ash_simulations.csv'
INFO: Copying 'input_data/eyja/conf_real/conf_a_priori.ini'
INFO: Subsitituting RUN_DIR => /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z
INFO: Subsitituting SCRIPT_DIR => /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/bin
INFO: Subsitituting TAG => EYJAFJALLA
INFO: Copying 'input_data/eyja/conf_real/conf_inversion.ini'
INFO: Copying 'input_data/eyja/conf_real/conf_match_files.ini'
INFO: Subsitituting SCRIPT_DIR => /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/bin
INFO: Copying 'input_data/eyja/conf_real/plume_heights.csv'
INFO: Copying '/home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/bin/inversion_job_environment.sh'
INFO: Subsitituting MEMORY => 1G
INFO: Subsitituting PARALLEL_ENVIRONMENT => NONE
INFO: Subsitituting PROJECT => NONE
INFO: Subsitituting QUEUE => NONE
INFO: Subsitituting RANDOM_SEED => 1647514333
INFO: Subsitituting RUN_DIR => /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z
INFO: Subsitituting RUNTIME => 1:30:0
INFO: Subsitituting SCRIPT_DIR => /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/bin
INFO: Subsitituting TAG => EYJAFJALLA
INFO: Subsitituting USER => jupyter-babrodtk
INFO: Renaming job script
 
################ SUMMARY #################
 
Job has been set up in /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z
Please check files in directory and then:
* Execute 'qsub /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z/job_script_EYJAFJALLA.sh' to submit job.
* Execute 'qdel <job id>' to delete job
* Execute 'watch qstat -j <jobid>' to see progress
 
############## END SUMMARY ###############
```
You can now take a look at the configuration files in the directory 
```
/home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z
```
(make sure you substitute this path with the actual path on your machine)
The contents of the folder should be something like the following:
```
EYJAFJALLA_20220317T1052Z
`-> a_priori_EYJAFJALLA.dat
`-> ash_simulations_EYJAFJALLA.csv  
`-> conf_inversion_EYJAFJALLA.ini
`-> job_script_EYJAFJALLA.sh
`-> ash_observations_EYJAFJALLA.csv
`-> conf_a_priori_EYJAFJALLA.ini
`-> conf_match_files_EYJAFJALLA.ini
`-> plume_heights_EYJAFJALLA.csv
```


## Running an inversion job
After the inversion job has been set up, and you have inspected the configuration files, you can run the job by submitting to your queueing system (e.g., qsub), or run directly on your computer as follows:
```
cd VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z
./job_script_EYJAFJALLA.sh
```
which should start the inversion procedure which takes approximately ten minutes and ends with something like the following:
```
...
=======================================
Writing output to /home/jupyter-babrodtk/VolcanicAsh/VolcanicAshInversion/output/EYJAFJALLA_20220317T1052Z/results/EYJAFJALLA_20220317T1259Z/inversion_009_0.10000000_a_posteriori.png
==== 20220317T1307Z: 'echo ==== Inversion run end ===='
==== Inversion run end ====
INFO: Deactivating Conda
INFO: Resetting path
```
After having completed this procedure, you should be able to inspect the results as PNG-files in the output directory. 