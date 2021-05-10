import numpy as np
import adios2
import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD
import sys
import pandas as pd
from multiprocessing import Pool
import glob
import os

fn = sys.argv[1]
ps = int(sys.argv[2])

def f(position):
    outlier_traj = mda.Universe(init_pdb, position)
    ref_traj = mda.Universe(ref_pdb_file)
    R = RMSD(outlier_traj, ref_traj, select = 'protein and name CA')
    R.run()
    return R.rmsd[:,2][0]

ref_pdb_file ='../MD_exps/fs-pep/pdb/1FME.pdb'

init_pdb = '../MD_exps/fs-pep/pdb/1FME-0.pdb'

pf = pd.DataFrame(columns=["fstep", "step", "R"])

with adios2.open(fn, "r") as fr:
    n = fr.steps()
    vars = fr.available_variables()
    
    print(vars)
    
    name = 'positions'
    shape = list(map(int, vars[name]['Shape'].split(",")))
    zs = list(np.zeros(len(shape), dtype=np.int))
    positions = fr.read(name, zs, shape, 0, n)
    print(type(positions))
    print(positions.shape)
    sys.stdout.flush()

    name = 'contact_map'
    shape = list(map(int, vars[name]['Shape'].split(",")))
    zs = list(np.zeros(len(shape), dtype=np.int))
    cms = fr.read(name, zs, shape, 0, n)
    print(type(cms))
    print(cms.shape)
    sys.stdout.flush()
    
    name = 'step'
    steps = fr.read(name, [],[], 0, n)
    print(type(steps))
    print(steps.shape)
    print(steps)
    sys.stdout.flush()

    name = 'velocities'
    shape = list(map(int, vars[name]['Shape'].split(",")))
    zs = list(np.zeros(len(shape), dtype=np.int))
    velocities = fr.read(name, zs, shape, 0, n)
    print(type(velocities))
    print(velocities.shape)
    sys.stdout.flush()


with Pool(processes=ps) as pool:
    Rs = pool.map(f, positions)

Rs1 = np.array(Rs)

index = Rs1 < 2.15

out_positions = positions[index]
out_contact_maps = cms[index]
out_Rs = Rs1[index]
out_velocities = velocities[index]

import utils
original_pdb = "../MD_exps/fs-pep/pdb/1FME-0.pdb"

for p, cm, v, r in zip(out_positions, out_contact_maps, out_velocities, out_Rs):
    fn1 = os.path.basename(fn).replace(".bp",f"_{r}.pdb")
    utils.write_pdb_frame(p, original_pdb, fn1)
    fn2 = os.path.basename(fn).replace(".bp",f"_{r}_v.npy")    
    np.save(fn2, v)
    fn3 = os.path.basename(fn).replace(".bp",f"_{r}_cm.npy")    
    np.save(fn3, cm)


