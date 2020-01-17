from __future__ import print_function

import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
import tftraj.rmsd as rmsd
import copy
from tensorboardX import SummaryWriter
from tf_rmsd import tf_centroid, tf_centroid_masked, tf_kabsch_rmsd_masked, tf_kabsch_rmsd
import pdb
import rmsd
import glob
import copy
import os
import shutil
import pickle as pkl

class Model(object):

    def __init__(self, data, n_max, dim_node, dim_edge, dim_h, dim_f, \
                batch_size, \
                mpnn_steps=5, alignment_type='default', tol=1e-5, \
                use_X=True, use_R=True, seed=0, \
                refine_steps=0, refine_mom=0.99, prior_T=1, num_cpus=8):
                
        self.cpu_config = tf.ConfigProto(
            device_count={"CPU": num_cpus, "GPU": 1},
            intra_op_parallelism_threads=num_cpus,
            inter_op_parallelism_threads=num_cpus,
            allow_soft_placement=True
        )

        # set random seed
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # hyper-parameters
        self.data = data
        self.mpnn_steps = mpnn_steps
        self.n_max, self.dim_node, self.dim_edge, self.dim_h, self.dim_f = n_max, dim_node, dim_edge, dim_h, dim_f
        #self.val_num_samples = val_num_samples
        self.tol = tol
        self.refine_steps = refine_steps
        self.refine_mom = refine_mom
        self.use_X = use_X
        self.use_R = use_R
        self.batch_size = batch_size
        self.b_size = tf.placeholder(dtype=tf.int32)

        if alignment_type == 'linear':
            self.msd_func = self.linear_transform_msd
        elif alignment_type == 'kabsch':
            self.msd_func = self.kabsch_msd
        elif alignment_type == 'default':
            self.msd_func = self.mol_msd
        self.sess = tf.Session(config=self.cpu_config)
        # variables
        self.G = tf.Graph()
        self.G.as_default()

        # allow the tensorflow graph to be flexible for number of samples in batch
        # will be useful for validation when we use multiple samples
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.dim_node], name="node")
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.n_max, 1], name="mask") # node yes = 1, no = 0
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.n_max, self.dim_edge], name="edge")
        self.pos = tf.placeholder(tf.float32, [self.batch_size, self.n_max, 3], name="pos")
        self.pos_to_proximity = self._pos_to_proximity(self.pos)
        self.proximity = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.n_max], name="proximity")
        mask = self.mask
        self.trn_flag = tf.placeholder(tf.bool, name="train_flag")

        # number of atoms
        self.n_atom = tf.reduce_sum( tf.transpose(self.mask, [0, 2, 1]), 2) #[batch_size, 1]

        # embedding for nodes/atoms
        self.node_embed = self._embed_node(self.node)
        
        # (20, 50, 50, 11) #(batch_size, n_max, n_max, dim_edge+1)
        self.edge_2 = tf.concat([self.edge, tf.tile( tf.reshape(self.n_atom, [self.batch_size, 1, 1, 1]), [1, self.n_max, self.n_max, 1] )], 3)

        # p(Z|G) -- prior of Z
        self.priorZ_edge_wgt = self._edge_nn(self.edge_2, name = 'priorZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.priorZ_hidden = self._MPNN(self.priorZ_edge_wgt, self.node_embed, name = 'priorZ', reuse = False)
        self.priorZ_out = self._g_nn(self.priorZ_hidden, self.node_embed, 2 * self.dim_h, name = 'priorZ', reuse = False, uname = 'priorZ_out')
        self.priorZ_mu, self.priorZ_lsgms = tf.split(self.priorZ_out, [self.dim_h, self.dim_h], 2)
        self.priorZ_sample = self._draw_sample(self.priorZ_mu, self.priorZ_lsgms, T=prior_T)

        # q(Z|R(X),G) -- posterior of Z, used R insted of X as input for simplicity, should be updated
        if use_R:
            self.postZ_edge_wgt = self._edge_nn(tf.concat([self.edge_2, tf.reshape(self.proximity, [self.batch_size, self.n_max, self.n_max, 1])], 3),  name = 'postZ', reuse = False)
        else:
            self.postZ_edge_wgt = self._edge_nn(self.edge_2,  name = 'postZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]

        if use_X:
            self.postZ_hidden = self._MPNN(self.postZ_edge_wgt, self._embed_node(tf.concat([self.node, self.pos], 2)), name = 'postZ', reuse = False)
        else:
            self.postZ_hidden = self._MPNN(self.postZ_edge_wgt, self.node_embed, name = 'postZ', reuse = False)

        self.postZ_out = self._g_nn(self.postZ_hidden, self.node_embed, 2 * self.dim_h, name = 'postZ', reuse = False, uname = 'postZ_out')
        self.postZ_mu, self.postZ_lsgms = tf.split(self.postZ_out, [self.dim_h, self.dim_h], 2)
        self.postZ_sample = self._draw_sample(self.postZ_mu, self.postZ_lsgms)

        # p(X|Z,G) -- posterior of X
        self.X_edge_wgt = self._edge_nn(self.edge_2, name = 'postX', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.X_hidden = self._MPNN(self.X_edge_wgt, self.postZ_sample + self.node_embed, name = 'postX', reuse = False)
        self.X_pred = self._g_nn(self.X_hidden, self.node_embed, 3, name = 'postX', reuse = False, mask=mask, uname = 'X_pred')
        
        # p(X|Z,G) -- posterior of X without sampling from latent space
        # used for iterative refinement of predictions
        # det stands for deterministic
        self.X_edge_wgt_det = self._edge_nn(self.edge_2, name = 'postX', reuse = True) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.X_hidden_det = self._MPNN(self.X_edge_wgt_det, self.postZ_mu + self.node_embed, name = 'postX', reuse = True)
        self.X_pred_det = self._g_nn(self.X_hidden_det, self.node_embed, 3, name = 'postX', reuse = True, mask=mask, uname = 'X_pred_det')

        # Prediction of X with p(Z|G) in the test phase
        self.PX_edge_wgt = self._edge_nn(self.edge_2, name = 'postX', reuse = True) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.PX_hidden = self._MPNN(self.PX_edge_wgt, self.priorZ_sample + self.node_embed, name = 'postX', reuse = True)
        self.PX_pred = self._g_nn(self.PX_hidden, self.node_embed, 3, name = 'postX', reuse = True, mask=mask, uname = 'PX_pred')

        self.saver = tf.train.Saver()


    def test(self, D1_v, D2_v, D3_v, D4_v, D5_v, MS_v, load_path = None, useFF=False, batch_size=10, val_num_samples=5):
                
        sess = tf.Session(config=self.cpu_config)
        if load_path is not None:
            print(load_path)
        saver = tf.train.import_meta_graph(load_path)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
        graph = tf.get_default_graph()
        refine_mom=0.99
        val_batch_size = int(batch_size / val_num_samples)
        n_batch_val = int(len(D1_v)/val_batch_size)
        val_size = D1_v.shape[0]
        valscores_mean = np.zeros(val_size)
        valscores_std = np.zeros(val_size)
        
        print(val_batch_size, n_batch_val, val_size)

        print ("testing model...")
        node = graph.get_tensor_by_name("node:0")
        mask = graph.get_tensor_by_name("mask:0")
        edge = graph.get_tensor_by_name("edge:0")
        trn_flag = graph.get_tensor_by_name("train_flag:0")
        pos = graph.get_tensor_by_name("pos:0")
        proximity = graph.get_tensor_by_name("proximity:0")
        pos_to_proximity = self._pos_to_proximity(pos)
        mask = graph.get_tensor_by_name("mask:0")
        X_pred = graph.get_tensor_by_name("g_nnpostX/X_pred_1:0")
        PX_pred = graph.get_tensor_by_name("g_nnpostX_2/PX_pred:0")

        b_size = tf.placeholder(dtype=tf.int32)
        use_X = False
        use_R = True
        valres=[]
        for i in range(n_batch_val):
            start_ = i * val_batch_size
            end_ = start_ + val_batch_size
            # input data is repeated to get more than 1 predictions for a molecule and take average prediction
            node_val = np.repeat(D1_v[start_:end_], val_num_samples, axis=0)
            mask_val = np.repeat(D2_v[start_:end_], val_num_samples, axis=0)
            edge_val = np.repeat(D3_v[start_:end_], val_num_samples, axis=0)
            proximity_val = np.repeat(D4_v[start_:end_], val_num_samples, axis=0)
            dict_val = {node: node_val, mask: mask_val, edge: edge_val, trn_flag: False, b_size: [batch_size]}
            D5_batch = sess.run(PX_pred, feed_dict=dict_val)
            # iterative refinement of posterior
            D5_batch_pred = copy.deepcopy(D5_batch)
            for r in range(self.refine_steps):
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
                res = self.getRMS(MS_v[ms_v_index], D5_batch_pred[j], useFF)
                valres.append(res)
            valres = np.array(valres)
            valres = np.reshape(valres, (val_batch_size, val_num_samples))
            valres_mean = np.mean(valres, axis=1)
            valres_std = np.std(valres, axis=1)
            valscores_mean[start_:end_] = valres_mean
            valscores_std[start_:end_] = valres_std
        print ("val scores: mean is {} , std is {}".format(np.mean(valscores_mean), np.mean(valscores_std)))

    def getRMS(self, prb_mol, ref_pos, useFF=False):
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


    def train(self, D1_t, D2_t, D3_t, D4_t, D5_t, MS_t, 
            load_path = None, save_path = None, w_reg=1e-3, epochs=10
        ):

        save_path = os.path.join('checkpoints/model.ckpt')
        print(save_path, flush=True)

        # objective functions
        cost_KLDZ = tf.reduce_mean( tf.reduce_sum( self._KLD(self.postZ_mu, self.postZ_lsgms, self.priorZ_mu, self.priorZ_lsgms), [1, 2]) ) # posterior | prior
        cost_KLD0 = tf.reduce_mean( tf.reduce_sum( self._KLD_zero(self.priorZ_mu, self.priorZ_lsgms), [1, 2]) ) # prior | N(0,1)

        mask = self.mask
        cost_X = tf.reduce_mean( self.msd_func(self.X_pred, self.pos, mask) )

        cost_op = cost_X + cost_KLDZ + w_reg * cost_KLD0 #hyperparameters!
        train_op = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(cost_op)

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

        # session
        n_batch = int(len(D1_t)/self.batch_size)
        np.set_printoptions(precision=5, suppress=True)

        # training
        print('::: start training for %d epochs' % epochs)
        num_epochs = epochs #2500
        valaggr_mean = np.zeros(num_epochs)
        valaggr_std = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            [D1_t, D2_t, D3_t, D4_t, D5_t] = self._permutation([D1_t, D2_t, D3_t, D4_t, D5_t])

            trnscores = np.zeros((n_batch, 4))
            for i in range(n_batch):
                print("Epoch: %d, batch: %d" % (epoch, i))
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size

                trnresult = self.sess.run([train_op, cost_op, cost_X, cost_KLDZ, cost_KLD0],
                                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_],
                                                 self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_],
                                                 self.pos: D5_t[start_:end_], self.trn_flag: True,
                                                 self.b_size: [self.batch_size]})
                trnresult = trnresult[1:]
                if debug:
                    print (i, n_batch)
                    print(trnresult, flush=True)

                # log results
                curr_iter = epoch * n_batch + i
                assert np.sum(np.isnan(trnresult)) == 0
                trnscores[i,:] = trnresult
            print(np.mean(trnscores,0), flush=True)

            if save_path is not None and not debug:
                self.saver.save(self.sess, save_path, global_step=num_epochs)
            # keep track of the best model as well in the separate checkpoint
            # it is done by copying the checkpoint
            if valaggr_mean[epoch] == np.min(valaggr_mean[0:epoch+1]) and not debug:
                for ckpt_f in glob.glob(save_path + '*'):
                    model_name_split = ckpt_f.split('/')
                    model_path = '/'.join(model_name_split[:-1])
                    model_name = model_name_split[-1]
                    best_model_name = model_name.split('.')[0] + '_best.' + '.'.join(model_name.split('.')[1:])
                    full_best_model_path = os.path.join(model_path, best_model_name)
                    full_model_path = ckpt_f
                    shutil.copyfile(full_model_path, full_best_model_path)
        self.sess.close()

    def do_mask(self, vec, m):
        return tf.boolean_mask(vec, tf.reshape(tf.greater(m, tf.constant(0.5)), [self.n_max,]) )

    def kabsch_msd(self, frames, targets, masks):
        losses = []
        for i in range(self.batch_size):
            frame = frames[i]
            target = targets[i]
            mask = masks[i]
            target_cent = target - tf_centroid_masked(target, mask, self.tol)
            frame_cent = frame - tf_centroid_masked(frame, mask, self.tol)
            losses.append(tf_kabsch_rmsd_masked(tf.stop_gradient(target_cent), frame_cent, mask, self.tol))

        loss = tf.stack(losses, 0)
        return loss

    def mol_msd(self, frames, targets, masks):
        frames -= tf.reduce_mean(frames, axis = 1, keepdims = True)
        targets -= tf.reduce_mean(targets, axis = 1, keepdims = True)

        loss = tf.stack([rmsd.squared_deviation( self.do_mask(frames[i], masks[i]), self.do_mask(targets[i], masks[i]) ) for i in range(self.batch_size)], 0)
        return loss / tf.reduce_sum(masks, axis=[1,2])

    def linear_transform_msd(self, frames, targets, masks):
        def linearly_transform_frames(padded_frames, padded_targets):
            s, u, v = tf.svd(padded_frames)
            tol = 1e-7
            atol = tf.reduce_max(s) * tol
            s = tf.boolean_mask(s, s > atol)
            s_inv = tf.diag(1. / s)
            pseudo_inverse = tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

            weight_matrix = tf.matmul(padded_targets, pseudo_inverse)
            transformed_frames = tf.matmul(weight_matrix, padded_frames)
            return transformed_frames

        padding = tf.constant([[0, 0], [0, 0], [0, 1]])
        padded_frames = tf.pad(frames, padding, 'constant', constant_values=1)
        padded_targets = tf.pad(targets, padding, 'constant', constant_values=1)

        mask_matrices = []
        for i in range(self.batch_size):
            mask_matrix = tf.diag(tf.reshape(masks[i], [-1]))
            mask_matrices.append(mask_matrix)
        #mask_matrix = tf.diag(tf.reshape(masks, [self.batch_size, -1]))
        mask_tensor = tf.stack(mask_matrices)
        masked_frames = tf.matmul(mask_tensor, padded_frames)
        masked_targets = tf.matmul(mask_tensor, padded_targets)
        transformed_frames = []
        for i in range(self.batch_size):
            transformed_frames.append(linearly_transform_frames(masked_frames[i], masked_targets[i]))
        transformed_frames = tf.stack(transformed_frames)
        #transformed_frames = linearly_transform_frames(masked_frames, masked_targets)
        loss = tf.losses.mean_squared_error(transformed_frames, masked_targets)

        return loss

    def _permutation(self, set):

        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]

        return set


    def _draw_sample(self, mu, lsgms, T=1):

        epsilon = tf.random_normal(tf.shape(lsgms), 0., 1.)
        sample = tf.multiply(tf.exp(0.5 * lsgms) * T, epsilon)
        sample = tf.add(mu, sample)
        sample = tf.multiply(sample, self.mask)

        return sample


    def _embed_node(self, inp): #[batch_size, n_max, dim_node]

        inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])

        inp = tf.layers.dense(inp, self.dim_h, activation = tf.nn.sigmoid)
        inp = tf.layers.dense(inp, self.dim_h, activation = tf.nn.tanh)

        inp = tf.reshape(inp, [self.batch_size, self.n_max, self.dim_h])
        inp = tf.multiply(inp, self.mask)

        return inp


    def _edge_nn(self, inp, name='', reuse=True): #[batch_size, n_max, n_max, dim_edge]

        with tf.variable_scope('edge_nn'+name, reuse=reuse):

            inp = tf.reshape(inp, [self.batch_size * self.n_max * self.n_max, int(inp.shape[3])])

            inp = tf.layers.dense(inp, 2 * self.dim_h, activation = tf.nn.sigmoid)
            inp = tf.layers.dense(inp, self.dim_h * self.dim_h, activation = tf.nn.tanh)

            inp = tf.reshape(inp, [self.batch_size, self.n_max, self.n_max, self.dim_h, self.dim_h])

        return inp


    def _msg_nn(self, wgt, node, name='', reuse=True):

        wgt = tf.reshape(wgt, [self.batch_size * self.n_max, self.n_max * self.dim_h, self.dim_h])
        node = tf.reshape(node, [self.batch_size * self.n_max, self.dim_h, 1])

        msg = tf.matmul(wgt, node)
        msg = tf.reshape(msg, [self.batch_size, self.n_max, self.n_max, self.dim_h])
        msg = tf.transpose(msg, perm = [0, 2, 3, 1])
        msg = tf.reduce_mean(msg, 3) / self.n_max

        return msg


    def _update_GRU(self, msg, node, name='', reuse=True, mask=None):

        if mask is None: mask=self.mask
        with tf.variable_scope('update_GRU'+name, reuse=reuse):

            msg = tf.reshape(msg, [self.batch_size * self.n_max, 1, self.dim_h])
            node = tf.reshape(node, [self.batch_size * self.n_max, self.dim_h])

            cell = tf.nn.rnn_cell.GRUCell(self.dim_h)
            _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)

            node_next = tf.reshape(node_next, [self.batch_size, self.n_max, self.dim_h])
            node_next = tf.multiply(node_next, mask)

        return node_next


    def _MPNN(self, edge_wgt, node_hidden_0, name='', reuse=True, true_mask=False):

        for i in range(self.mpnn_steps): #hyperparameters!

            mv_0 = self._msg_nn(edge_wgt, node_hidden_0)
            if true_mask and i == self.mpnn_steps - 1:
                node_hidden_0 = self._update_GRU(mv_0, node_hidden_0, name=name, reuse=(i + reuse) != 0, mask=self.true_masks)
            else:
                node_hidden_0 = self._update_GRU(mv_0, node_hidden_0, name=name, reuse=(i+reuse)!=0)#[batch_size, n_max, dim_h]

        return node_hidden_0


    def _g_nn(self, inp, node, outdim, name='', reuse=True, mask=None, uname=''): #[batch_size, n_max, -]

        if mask is None: mask = self.mask
        with tf.variable_scope('g_nn'+name, reuse=reuse):

            inp = tf.concat([inp, node], 2)

            inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])
            inp = tf.layers.dropout(inp, rate = 0.2, training = self.trn_flag)
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dropout(inp, rate = 0.2, training = self.trn_flag)
            #inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dense(inp, outdim)

            inp = tf.reshape(inp, [self.batch_size, self.n_max, outdim])
            inp = tf.multiply(inp, mask, name = uname)

        return inp


    def _pos_to_proximity(self, pos, reuse=True): #[batch_size, n_max, 3]

        with tf.variable_scope('pos_to_proximity', reuse=reuse):

            pos_1 = tf.expand_dims(pos, axis = 2)
            pos_2 = tf.expand_dims(pos, axis = 1)

            pos_sub = tf.subtract(pos_1, pos_2)
            proximity = tf.square(pos_sub)
            proximity = tf.reduce_sum(proximity, 3)
            proximity = tf.sqrt(proximity + 1e-5)

            proximity = tf.reshape(proximity, [self.batch_size, self.n_max, self.n_max])
            proximity = tf.multiply(proximity, self.mask)
            proximity = tf.multiply(proximity, tf.transpose(self.mask, perm = [0, 2, 1]))

            proximity = tf.matrix_set_diag(proximity, [[0] * self.n_max] * self.batch_size)

        return proximity


    def _KLD(self, mu0, lsgm0, mu1, lsgm1):# [batch_size, n_max, dim_h]

        var0 = tf.exp(lsgm0)
        var1 = tf.exp(lsgm1)
        a = tf.div( var0 + 1e-5, var1 + 1e-5)
        b = tf.div( tf.square( tf.subtract(mu1, mu0) ), var1 + 1e-5)
        c = tf.log( tf.div(var1 + 1e-5, var0 + 1e-5 ) + 1e-5)

        kld = 0.5 * tf.reduce_sum(a + b - 1 + c, 2, keepdims = True) * self.mask

        return kld


    def _KLD_zero(self, mu0, lsgm0):# [batch_size, n_max, dim_h]

        a = tf.exp(lsgm0) + tf.square(mu0)
        b = 1 + lsgm0

        kld = 0.5 * tf.reduce_sum(a - b, 2, keepdims = True) * self.mask

        return kld
