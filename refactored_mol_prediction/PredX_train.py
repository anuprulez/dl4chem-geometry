from __future__ import print_function

import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys, gc, os
import PredX_MPNN as MPNN
#import scipy
#from scipy import sparse
import argparse
import getpass
from test_tube import HyperOptArgumentParser, Experiment
from test_tube.hpc import SlurmCluster

def data_path():
    return "data/"

def train(args, exp=None):
    n_max = 50
    dim_node = 35
    dim_edge = 10
    ntst = 20
    dim_h = args.dim_h
    dim_f = args.dim_f
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples

    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)

    save_path = os.path.join(args.ckptdir, args.model_name + '_model.ckpt')
    molvec_fname = data_path() + args.data+'_molvec_'+str(n_max)+'.p'
    molset_fname = data_path() + args.data+'_molset_'+str(n_max)+'.p'

    print('::: load data')
    [D1, D2, D3, D4, D5] = pkl.load(open(molvec_fname,'rb'))
    D1 = D1.todense()
    D2 = D2.todense()
    D3 = D3.todense()

    ntrn = len(D5)-ntst
    [molsup, molsmi] = pkl.load(open(molset_fname,'rb'))

    D1_trn = D1[:ntrn]
    D2_trn = D2[:ntrn]
    D3_trn = D3[:ntrn]
    D4_trn = D4[:ntrn]
    D5_trn = D5[:ntrn]
    molsup_trn =molsup[:ntrn]
    D1_tst = D1[ntrn:ntrn+ntst]
    D2_tst = D2[ntrn:ntrn+ntst]
    D3_tst = D3[ntrn:ntrn+ntst]
    D4_tst = D4[ntrn:ntrn+ntst]
    D5_tst = D5[ntrn:ntrn+ntst]
    molsup_tst =molsup[ntrn:ntrn+ntst]
    print ('::: num train samples is ')
    print(D1_trn.shape, D3_trn.shape)
    print ('::: num test samples is ')
    print(D1_tst.shape, D3_tst.shape)

    tm_trn, tm_val, tm_tst = None, None, None

    del D1, D2, D3, D4, D5, molsup

    model = MPNN.Model(args.data, n_max, dim_node, dim_edge, dim_h, dim_f, batch_size, \
                        mpnn_steps=args.mpnn_steps, alignment_type=args.alignment_type, tol=args.tol,\
                        use_X=args.use_X, use_R=args.use_R, seed=args.seed, \
                        refine_steps=args.refine_steps, refine_mom=args.refine_mom, \
                        prior_T=args.prior_T)

    with model.sess:
        '''
        model.train(D1_trn, D2_trn, D3_trn, D4_trn, D5_trn, molsup_trn, \
                        load_path=args.loaddir, save_path=save_path, \
                        w_reg=args.w_reg, epochs=args.num_epochs)
        ):
        '''
        model.test(D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst, \
                            load_path=args.loaddir, \
                            useFF=args.useFF, batch_size=args.batch_size, val_num_samples=args.val_num_samples)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--data', type=str, default='COD', choices=['COD', 'QM9', 'CSD'])
    parser.add_argument('--ckptdir', type=str, default='checkpoints/')
    parser.add_argument('--eventdir', type=str, default='events/')
    parser.add_argument('--savepreddir', type=str, default=None,
                        help='path where predictions of the network are save')
    parser.add_argument('--savepermol', action='store_true', help='save results per molecule')
    parser.add_argument('--loaddir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='neuralnet')
    parser.add_argument('--alignment_type', type=str, default='kabsch', choices=['default', 'linear', 'kabsch'])
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--use_val', action='store_true', default=False, help='use validation set')
    parser.add_argument('--seed', type=int, default=1334, help='random seed for experiments')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--val_num_samples', type=int, default=5,
                        help='number of samples from prior used for validation')
    parser.add_argument('--tol', type=float, default=1e-5, help='tolerance for masking used in svd calculation')
    parser.add_argument('--prior_T', type=float, default=1, help='temperature to use for the prior')
    parser.add_argument('--use_X', action='store_true', default=False, help='use X as input for posterior of Z')
    parser.add_argument('--use_R', action='store_true', default=True, help='use R(X) as input for posterior of Z')
    parser.add_argument('--w_reg', type=float, default=1e-5, help='weight for conditional prior regularization')
    parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
    parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
    parser.add_argument('--useFF', action='store_true', help='use force field minimisation if testing')

    parser.add_argument('--dim_h', type=int, default=50, help='dimension of the hidden')
    parser.add_argument('--dim_f', type=int, default=100, help='dimension of the hidden')
    parser.add_argument('--mpnn_steps', type=int, default=5, help='number of mpnn steps')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of training steps')

    args = parser.parse_args()

    train(args)
