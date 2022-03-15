import pytest

import subprocess
import os
import json
import time
import numpy as np
from netCDF4 import Dataset
from scipy import sparse

#Set common paths
testpath = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.abspath(os.path.join(testpath, "data"))
basepath = os.path.abspath(os.path.join(testpath, "..", ".."))

my_env = os.environ
my_env["PYTHONPATH"] = my_env.get("PYTHONPATH", "") + ":" + basepath

#Set filenames used by multiple tests
hybrid_levels = os.path.join(datapath, "Vertical_levels_22_650m.txt")
plume_heights = os.path.join(datapath, "plume_heights.csv")
matched_files = os.path.join(datapath, "matched_files.csv")
a_priori_file = os.path.join(datapath, "a_priori.json")
inversion_matrix = os.path.join(datapath, "inversion_system_matrix.npz")
simulation_csv = os.path.join(datapath, "ash_simulations.csv")
observation_csv = os.path.join(datapath, "ash_observations.csv")
matched_files = os.path.join(datapath, "matched_files.csv")


def file_age(filename):
    return time.time() - os.path.getmtime(filename)


@pytest.mark.dependency()
def test_APriori():
    """ Regression test for a priori generation """
    
    script = os.path.join(basepath, "AshInv", "APriori.py")
    reference = os.path.join(datapath, "a_priori_reference.json")
    
    for f in [a_priori_file]:
        if os.path.exists(f):
            os.remove(f)
    
    subprocess.check_call([script, 
                           "--volcano_altitude", "1666", 
                           "--hybrid_levels_file", hybrid_levels,
                           "--plume_heights_file", plume_heights,
                           "--a_priori_file", a_priori_file],
                         env=my_env,
                         cwd=testpath)
    
    assert file_age(a_priori_file) < 10
    
    #Read a priori and reference
    with open(reference, 'r') as infile:
        a_priori_reference = json.load(infile)
    with open(a_priori_file, 'r') as infile:
        a_priori = json.load(infile)
        
    #Compare important variables
    keys = ['level_boundaries', 'level_heights', 'volcano_altitude', 'a_priori_2d', 'a_priori_2d_uncertainty']
    for key in keys:
        assert np.allclose(a_priori[key], a_priori_reference[key])


        
        
        
@pytest.mark.dependency()
def test_MatchFiles():
    """ Regression test for matching files """
    
    script = os.path.join(basepath, "AshInv", "MatchFiles.py")
    output_file = os.path.join(datapath, 'MSG2-ASH-IcelandEurope_1088_0088_1537x494-201004141200_ExtendedVOLE_lognormal_1_5_gridded_matched.nc')
    output_file_txt = os.path.join(datapath, 'MSG2-ASH-IcelandEurope_1088_0088_1537x494-201004141200_ExtendedVOLE_lognormal_1_5_gridded_matched.txt')
    reference = os.path.join(datapath, 'MSG2-ASH-IcelandEurope_1088_0088_1537x494-201004141200_ExtendedVOLE_lognormal_1_5_gridded_matched_reference.nc')
    
    for f in [output_file, output_file_txt, matched_files]:
        if os.path.exists(f):
            os.remove(f)
    
    subprocess.check_call([script, 
                           "--zero_thinning", "0.75", 
                           "--obs_zero", "1.0e-5", 
                           "--obs_flag_max", "1.95", 
                           "--simulation_csv", simulation_csv,
                           "--observation_csv", observation_csv,
                           "--simulation_basedir", datapath,
                           "--observation_basedir", datapath,
                           "--output_dir", datapath],
                         env=my_env,
                         cwd=testpath)
    
    assert file_age(output_file) < 10
    
    #Read a priori and reference
    try:
        nc_file = Dataset(output_file, 'r')
        nc_file_reference = Dataset(reference, 'r')
        
        lon=nc_file['longitude'][:]
        lat=nc_file['latitude'][:]
        obs=nc_file['obs'][:]
        obs_flag=nc_file['obs_flag'][:]
        sim=nc_file['sim'][:]
        
        lon_ref=nc_file['longitude'][:]
        lat_ref=nc_file['latitude'][:]
        obs_ref=nc_file['obs'][:]
        obs_flag_ref=nc_file['obs_flag'][:]
        sim_ref=nc_file['sim'][:]
    except:
        raise
    finally:
        nc_file.close()

    assert np.allclose(lon, lon_ref)
    assert np.allclose(lat, lat_ref)
    assert np.allclose(obs, obs_ref)
    assert np.allclose(obs_flag, obs_flag_ref)
    assert np.allclose(sim, sim_ref)
    assert obs.max() > 0
    assert sim.max() > 0

    
    
    
@pytest.mark.dependency(depends=["test_APriori", "test_MatchFiles"])
def test_AshInversion():
    """ Regression test for inversion procedure """
    
    script = os.path.join(basepath, "AshInv", "AshInversion.py")
    a_posteriori = os.path.join(datapath, 'inversion_000_1.00000000_a_posteriori.json')
    reference = os.path.join(datapath, 'inversion_000_1.00000000_a_posteriori_reference.json')
    
    for f in [inversion_matrix, a_posteriori]:
        if os.path.exists(f):
            os.remove(f)
    
    subprocess.check_call([script, 
                           "--matched_files", matched_files,
                           "--a_priori_file", a_priori_file,
                           "--output_dir", datapath,
                           "--scale_a_priori", "1.0e-9", 
                           "--scale_emission", "1.0e-9", 
                           "--scale_observation", "1.0e-3", 
                           "--smoothing_epsilon", "1.0e-4",
                           "--a_priori_epsilon", "1.0",
                           "--max_iter", "50",
                           "--store_full_matrix"],
                         env=my_env,
                         cwd=testpath)
    
    assert file_age(a_posteriori) < 10

    #Read a priori and reference
    with open(reference, 'r') as infile:
        a_priori_reference = json.load(infile)
    with open(a_posteriori, 'r') as infile:
        a_priori = json.load(infile)
        
    #Compare important variables
    keys = ['a_posteriori_2d', 'level_heights', 'volcano_altitude', 'a_priori_2d']
    for key in keys:
        assert np.allclose(a_priori[key], a_priori_reference[key])

        
        
        
        
@pytest.mark.dependency(depends=["test_AshInversion"])
def test_AshInversionMatrix():
    """ Test that the fast ash inversion matrix is computed correctly """
    
    inversion_matrix = os.path.join(datapath, "inversion_system_matrix.npz")
    
    with np.load(inversion_matrix) as data:
        M_data = data['M_data']
        M_indices = data['M_indices']
        M_indptr = data['M_indptr']
        M_shape = data['M_shape']
        sigma_o = data['sigma_o']
        Q = data['Q']
        
    M = sparse.csr_matrix((M_data, M_indices, M_indptr), shape=M_shape)
    
    sigma_o_m2 = sigma_o**-2
    N = sigma_o.size
    indices = np.arange(0, N)
    indptr = np.append(indices, indices[-1])
    sigma_o_m2 = sparse.csr_matrix((sigma_o_m2, indices, indptr), shape=(N, N))
    
    assert Q.max() > 0
    
    Q_star = (M.T).dot(sigma_o_m2)
    Q_new = Q_star.dot(M).todense()
        
    assert np.allclose(Q, Q_new)
    