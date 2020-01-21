import pickle as pkl
import os
import PredX_MPNN as MPNN
import argparse
import zipfile


def zipCompress(path, ziph):
    '''
    Compress tensorflow model files to zip
    '''
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(path, file), file, compress_type=zipfile.ZIP_DEFLATED)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train(args, exp=None):
    '''
    Train on molecules and create a trained model
    '''
    n_max = args.n_max
    dim_node = args.dim_node
    dim_edge = args.dim_edge
    ntst = args.test_size
    dim_h = args.dim_h
    dim_f = args.dim_f
    batch_size = args.batch_size

    create_dir(args.ckptdir)
    # load data
    save_path = os.path.join(args.ckptdir, args.model_name)

    print('::: load data')
    [D1, D2, D3, D4, D5] = pkl.load(open(args.train_molvec, 'rb'))
    D1 = D1.todense()
    D2 = D2.todense()
    D3 = D3.todense()

    ntst = int(D1.shape[0] * ntst)

    ntrn = len(D5) - ntst
    [molsup, molsmi] = pkl.load(open(args.train_molset, 'rb'))

    # divide into train and test sets
    # train data
    D1_trn = D1[:ntrn]
    D2_trn = D2[:ntrn]
    D3_trn = D3[:ntrn]
    D4_trn = D4[:ntrn]
    D5_trn = D5[:ntrn]
    molsup_trn = molsup[:ntrn]
    # test data
    D1_tst = D1[ntrn:ntrn+ntst]
    D2_tst = D2[ntrn:ntrn+ntst]
    D3_tst = D3[ntrn:ntrn+ntst]
    D4_tst = D4[ntrn:ntrn+ntst]
    D5_tst = D5[ntrn:ntrn+ntst]
    molsup_tst = molsup[ntrn:ntrn+ntst]

    # set aside test data
    with open(args.test_file, 'wb') as f:
        pkl.dump([D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst], f)

    print('::: size of training data ')
    print(D1_trn.shape, D3_trn.shape)

    print('::: size of test data')
    print(D1_tst.shape, D3_tst.shape)

    del D1, D2, D3, D4, D5, molsup

    model = MPNN.Model(n_max, dim_node, dim_edge, dim_h, dim_f, batch_size,
                        mpnn_steps=args.mpnn_steps, alignment_type=args.alignment_type, tol=args.tol,
                        use_X=args.use_X, use_R=args.use_R, seed=args.seed,
                        refine_steps=args.refine_steps, refine_mom=args.refine_mom,
                        prior_T=args.prior_T)

    with model.sess:
        model.train(D1_trn, D2_trn, D3_trn, D4_trn, D5_trn, molsup_trn, load_path=save_path, w_reg=args.w_reg, epochs=args.num_epochs)
        print("::: finished training")
        # compress model files into one zipped file
        create_dir(os.path.dirname(args.zipdir))
        zipf = zipfile.ZipFile(args.zipdir, 'w')
        zipCompress(os.path.dirname(save_path), zipf)
        zipf.close()
        print("::: zipped model at %s" % args.zipdir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--train_molvec', type=str, help='molvec file for training')
    parser.add_argument('--train_molset', type=str, help='molset file for training')
    parser.add_argument('--test_file', type=str, help='file for testing')
    parser.add_argument('--zipdir', type=str, help="zip file for compressed model")
    parser.add_argument('--ckptdir', type=str, help="directory to keep the model checkpoints")
    parser.add_argument('--model_name', type=str, help="comman name for checkpoint model files")
    parser.add_argument('--test_size', type=float, help='size of test data')

    # training parameters
    parser.add_argument('--alignment_type', type=str, default='kabsch', choices=['default', 'linear', 'kabsch'])
    parser.add_argument('--seed', type=int, default=1334, help='random seed for experiments')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--tol', type=float, default=1e-5, help='tolerance for masking used in svd calculation')
    parser.add_argument('--prior_T', type=float, default=1, help='temperature to use for the prior')
    parser.add_argument('--use_X', action='store_true', default=False, help='use X as input for posterior of Z')
    parser.add_argument('--use_R', action='store_true', default=True, help='use R(X) as input for posterior of Z')
    parser.add_argument('--w_reg', type=float, default=1e-5, help='weight for conditional prior regularization')
    parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
    parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
    parser.add_argument('--useFF', default=True, action='store_true', help='use force field minimisation if testing')
    parser.add_argument('--dim_h', type=int, default=50, help='dimension of the hidden')
    parser.add_argument('--dim_f', type=int, default=100, help='dimension of the hidden')
    parser.add_argument('--n_max', type=int, default=50, help='maximum number of atoms')
    parser.add_argument('--dim_node', type=int, default=35, help='dimension of the nodes')
    parser.add_argument('--dim_edge', type=int, default=10, help='dimension of the edges')
    parser.add_argument('--mpnn_steps', type=int, default=5, help='number of mpnn steps')
    parser.add_argument('--num_epochs', type=int, default=2500, help='number of training steps')

    args = parser.parse_args()

    train(args)
