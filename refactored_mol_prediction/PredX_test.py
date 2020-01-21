import pickle as pkl
import numpy as np
import argparse
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import os
import zipfile


def getRMS(prb_mol, ref_pos, useFF=False):
    '''
    Compute the root mean square distance
    between true and predicted conformation
    '''
    def optimizeWithFF(mol):

        molf = Chem.AddHs(mol, addCoords=True)
        AllChem.MMFFOptimizeMolecule(molf)
        molf = Chem.RemoveHs(molf)

        return molf

    n_est = prb_mol.GetNumAtoms()
    ref_cf = Chem.rdchem.Conformer(n_est)
    for k in range(n_est):
        ref_cf.SetAtomPosition(k, ref_pos[k].tolist())

    ref_mol = copy.deepcopy(prb_mol)
    ref_mol.RemoveConformer(0)
    # add predicted conformation to molecule
    ref_mol.AddConformer(ref_cf)
    # compute error
    if useFF:
        try:
            res = AllChem.AlignMol(prb_mol, optimizeWithFF(ref_mol))
        except:
            res = AllChem.AlignMol(prb_mol, ref_mol)
    else:
        res = AllChem.AlignMol(prb_mol, ref_mol)

    return res


def _pos_to_proximity(pos, batch_size, n_max, mask, reuse=True): #[batch_size, n_max, 3]

    with tf.variable_scope('pos_to_proximity', reuse=reuse):

        pos_1 = tf.expand_dims(pos, axis = 2)
        pos_2 = tf.expand_dims(pos, axis = 1)

        pos_sub = tf.subtract(pos_1, pos_2)
        proximity = tf.square(pos_sub)
        proximity = tf.reduce_sum(proximity, 3)
        proximity = tf.sqrt(proximity + 1e-5)

        proximity = tf.reshape(proximity, [batch_size, n_max, n_max])
        proximity = tf.multiply(proximity, mask)
        proximity = tf.multiply(proximity, tf.transpose(mask, perm = [0, 2, 1]))

        proximity = tf.matrix_set_diag(proximity, [[0] * n_max] * batch_size)

    return proximity


def extract_model(load_path):
    '''
    Extract files from model
    '''
    dir_name = os.path.dirname(load_path)
    with zipfile.ZipFile(load_path, 'r') as zip_ref:
        zip_ref.extractall(dir_name)


def set_cpu(num_cpus):
    cpu_config = tf.ConfigProto(
        device_count={"CPU": num_cpus, "GPU": 1},
        intra_op_parallelism_threads=num_cpus,
        inter_op_parallelism_threads=num_cpus,
        allow_soft_placement=True
    )
    return cpu_config


def predict(args):
    '''
    Predict position of atoms for a new molecule
    '''
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples
    num_cpus = args.num_cpus
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples
    useFF = args.useFF
    refine_steps = args.refine_steps
    refine_mom = args.refine_mom
    use_X = args.use_X
    use_R = args.use_R
    load_path = args.path_model
    test_file = args.test_file

    print('::: load data')
    [D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst] = pkl.load(open(test_file, 'rb'))

    # maximum number of atoms
    n_max = D1_tst.shape[1]

    print('::: size of test data ')
    print(D1_tst.shape, D3_tst.shape)
    print('::: num of molecules is %d' % D1_tst.shape[0])
    print("::: load model " + load_path)

    extract_model(load_path)

    tf.reset_default_graph()
    sess = tf.Session(config=set_cpu(num_cpus))
    with sess:
        trained_model_path = "{}/{}".format(os.path.dirname(load_path), args.model_name)
        # import graph
        saver = tf.train.import_meta_graph(trained_model_path + ".meta")
        # restore model session
        saver.restore(sess, trained_model_path)
        graph = tf.get_default_graph()

        val_batch_size = int(batch_size/val_num_samples)
        n_batch_val = int(len(D1_tst)/val_batch_size)
        val_size = D1_tst.shape[0]
        valscores_mean = np.zeros(val_size)
        valscores_std = np.zeros(val_size)

        print("::: testing model...")
        # get the variable names from the loaded tensorflow graph
        node = graph.get_tensor_by_name("node:0")
        mask = graph.get_tensor_by_name("mask:0")
        edge = graph.get_tensor_by_name("edge:0")
        trn_flag = graph.get_tensor_by_name("train_flag:0")
        pos = graph.get_tensor_by_name("pos:0")
        proximity = graph.get_tensor_by_name("proximity:0")
        pos_to_proximity = _pos_to_proximity(pos, batch_size, n_max, mask)
        X_pred = graph.get_tensor_by_name("g_nnpostX/X_pred:0")
        PX_pred = graph.get_tensor_by_name("g_nnpostX_2/PX_pred:0")
        # define batch size variable
        b_size = tf.placeholder(dtype=tf.int32)
        for i in range(n_batch_val):
            valres = list()
            start_ = i * val_batch_size
            end_ = start_ + val_batch_size
            # input data is repeated to get more than 1 predictions for a molecule and take average prediction
            node_val = np.repeat(D1_tst[start_:end_], val_num_samples, axis=0)
            mask_val = np.repeat(D2_tst[start_:end_], val_num_samples, axis=0)
            edge_val = np.repeat(D3_tst[start_:end_], val_num_samples, axis=0)
            dict_val = {node: node_val, mask: mask_val, edge: edge_val, trn_flag: False, b_size: [batch_size]}
            D5_batch = sess.run(PX_pred, feed_dict=dict_val)
            # iterative refinement of posterior
            D5_batch_pred = copy.deepcopy(D5_batch)
            print("::: refining posterior")
            for r in range(refine_steps):
                if use_X:
                    dict_val[pos] = D5_batch_pred
                if use_R:
                    pred_proximity = sess.run(pos_to_proximity, feed_dict={pos: D5_batch_pred, mask: mask_val, b_size: [batch_size]})
                    dict_val[proximity] = pred_proximity
                D5_batch = sess.run(X_pred, feed_dict=dict_val)
                D5_batch_pred = refine_mom * D5_batch_pred + (1-refine_mom) * D5_batch
            print("::: computing RMSD")
            for j in range(D5_batch_pred.shape[0]):
                ms_v_index = int(j / val_num_samples) + start_
                # compute error between true and predicted conformations
                # 'D5_batch_pred' variable contains the predicted positions of atoms in a molecule in 3D space
                res = getRMS(molsup_tst[ms_v_index], D5_batch_pred[j], useFF)
                valres.append(res)
            valres = np.array(valres)
            valres = np.reshape(valres, (val_batch_size, val_num_samples))
            valres_mean = np.mean(valres, axis=1)
            valres_std = np.std(valres, axis=1)
            print("::: batch {} test scores: mean RMSD is {}, standard deviation (RMSD) is {}".format(i + 1, valres_mean[0], valres_std[0]))
            valscores_mean[start_:end_] = valres_mean
            valscores_std[start_:end_] = valres_std
        print()
        print("Overall test scores: mean RMSD is {}, standard deviation (RMSD) is {}".format(np.mean(valscores_mean), np.mean(valscores_std)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test network')
    
    parser.add_argument('--test_file', type=str, help="file for testing")
    parser.add_argument('--path_model', type=str, help="path to the zipped model")
    parser.add_argument('--model_name', type=str, help='base name for the model')
    
    # the following parameters should not be changed
    # ---------------------------------------------------
    # the value of batch_size parameter should be same as used during training
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    # the number of samples drawn for each molecule
    parser.add_argument('--val_num_samples', type=int, default=10, help='number of samples from prior used for validation')
    parser.add_argument('--use_X', action='store_true', default=False, help='use X as input for posterior of Z')
    parser.add_argument('--use_R', action='store_true', default=True, help='use R(X) as input for posterior of Z')
    parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
    parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
    parser.add_argument('--useFF', default=True, action='store_true', help='use force field minimisation if testing')
    # -----------------------------------------------------
    # number of CPUs. Can be changed depending on the availability
    parser.add_argument('--num_cpus', default=8, help='number of CPUs')

    args = parser.parse_args()

    predict(args)
