# Molecular Geometry Prediction (refactored)

## How to set up the environment

1. `conda create --name molecule_pred`
2. `conda install python=3.6 tensorflow=1.12 rdkit=2018.09.1 scikit-learn=0.21.3`
3. `pip install rmsd==1.3.2`
4. `pip install zipfile36==0.1.3`

## How to execute the scripts

1. Create features from molecular data:
    - Execute `sh run_cod_featurize.sh`. Change the number of molecules by setting the parameter `num_mol`

2. Train on the molecular features:
    - Execute `sh run_training.sh`. Change the number of training iterations and the size of test data by setting parameters `num_epochs` and `test_size` respectively.
    
3. Test/predict the 3D position of atoms in test molecules:
    - Execute `sh run_test.sh`
