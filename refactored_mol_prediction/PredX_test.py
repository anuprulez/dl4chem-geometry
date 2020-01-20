from __future__ import print_function

import pickle as pkl
import numpy as np
import argparse
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import os

    
def getRMS(prb_mol, ref_pos, useFF=False):
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
    ref_mol.AddConformer(ref_cf)

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

def test(args, exp=None):

    n_max = 50
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples
    num_cpus = args.num_cpus
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples
    useFF = args.useFF
    refine_steps = args.refine_steps
    refine_mom = float(args.refine_mom)
    use_X = args.use_X
    use_R = args.use_R
    load_path = args.loaddir
    test_file = args.testfile
    
    cpu_config = tf.ConfigProto(
        device_count={"CPU": num_cpus, "GPU": 1},
        intra_op_parallelism_threads=num_cpus,
        inter_op_parallelism_threads=num_cpus,
        allow_soft_placement=True
    )

    print('::: load data')
    [D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst] = pkl.load(open(test_file,'rb'))

    print ('::: num test samples is ')
    print(D1_tst.shape, D3_tst.shape)
    
    tf.reset_default_graph()

    print(":: load model " + load_path)
    
    sess = tf.Session(config=cpu_config)
    with sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        graph = tf.get_default_graph()

        val_batch_size = int(batch_size/val_num_samples)
        n_batch_val = int(len(D1_tst)/val_batch_size)
        val_size = D1_tst.shape[0]
        valscores_mean = np.zeros(val_size)
        valscores_std = np.zeros(val_size)
    
        print ("testing model...")
        node = graph.get_tensor_by_name("node:0")
        mask = graph.get_tensor_by_name("mask:0")
        edge = graph.get_tensor_by_name("edge:0")
        trn_flag = graph.get_tensor_by_name("train_flag:0")
        pos = graph.get_tensor_by_name("pos:0")
        proximity = graph.get_tensor_by_name("proximity:0")
        pos_to_proximity = _pos_to_proximity(pos, batch_size, n_max, mask)
        X_pred = graph.get_tensor_by_name("g_nnpostX/X_pred:0")
        PX_pred = graph.get_tensor_by_name("g_nnpostX_2/PX_pred:0")

        b_size = tf.placeholder(dtype=tf.int32)
        use_X = False
        use_R = True
        valres=[]
        for i in range(n_batch_val):
            start_ = i * val_batch_size
            end_ = start_ + val_batch_size
            # input data is repeated to get more than 1 predictions for a molecule and take average prediction
            node_val = np.repeat(D1_tst[start_:end_], val_num_samples, axis=0)
            mask_val = np.repeat(D2_tst[start_:end_], val_num_samples, axis=0)
            edge_val = np.repeat(D3_tst[start_:end_], val_num_samples, axis=0)
            proximity_val = np.repeat(D4_tst[start_:end_], val_num_samples, axis=0)
            dict_val = {node: node_val, mask: mask_val, edge: edge_val, trn_flag: False, b_size: [batch_size]}
            D5_batch = sess.run(PX_pred, feed_dict=dict_val)
            # iterative refinement of posterior
            D5_batch_pred = copy.deepcopy(D5_batch)
            for r in range(refine_steps):
                if use_X:
                    dict_val[pos] = D5_batch_pred
                if use_R:
                    pred_proximity = sess.run(pos_to_proximity, \
                                    feed_dict={pos: D5_batch_pred, \
                                                mask:mask_val, b_size: [batch_size]})
                    dict_val[proximity] = pred_proximity
                D5_batch = sess.run(X_pred, feed_dict=dict_val)
                D5_batch_pred = \
                refine_mom * D5_batch_pred + (1-refine_mom) * D5_batch
            valres=[]
            for j in range(D5_batch_pred.shape[0]):
                ms_v_index = int(j / val_num_samples) + start_
                res = getRMS(molsup_tst[ms_v_index], D5_batch_pred[j], useFF)
                valres.append(res)
            valres = np.array(valres)
            valres = np.reshape(valres, (val_batch_size, val_num_samples))
            valres_mean = np.mean(valres, axis=1)
            valres_std = np.std(valres, axis=1)
            valscores_mean[start_:end_] = valres_mean
            valscores_std[start_:end_] = valres_std
        print ("val scores: mean is {} , std is {}".format(np.mean(valscores_mean), np.mean(valscores_std)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--testfile', type=str, default='data/mol_test.p')
    parser.add_argument('--loaddir', type=str, default='checkpoints/')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--val_num_samples', type=int, default=5,
                        help='number of samples from prior used for validation')
    parser.add_argument('--use_X', action='store_true', default=False, help='use X as input for posterior of Z')
    parser.add_argument('--use_R', action='store_true', default=True, help='use R(X) as input for posterior of Z')
    parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
    parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
    parser.add_argument('--useFF', action='store_true', help='use force field minimisation if testing')
    parser.add_argument('--num_cpus', default=8, help='number of CPUs')

    args = parser.parse_args()

    test(args)
