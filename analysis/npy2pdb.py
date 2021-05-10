import utils
import sys
import numpy as np

original_pdb = "../MD_exps/fs-pep/pdb/1FME.pdb"
output_pdb_fn = "test.pdb"

input = np.load(sys.argv[1])

print(type(input))
print(input.shape)

in1 = input[0]

utils.write_pdb_frame(in1, original_pdb, output_pdb_fn)
