import pickle as pk
from ToolScripts.TimeLogger import log
import torch as t
import scipy.sparse as sp
import numpy as np
import os
import argparse


def mkdir(dataset):
    DIR = os.path.join(os.getcwd(), "History", dataset)
    if not os.path.exists(DIR):
        os.makedirs(DIR)

def loadData2(datasetStr, cv):
    assert datasetStr == "Tianchi_time"
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    with open(DIR + '/pvTime.csv'.format(cv), 'rb') as fs:
        pvTimeMat = pk.load(fs)
    with open(DIR + '/cartTime.csv'.format(cv), 'rb') as fs:
        cartTimeMat = pk.load(fs)
    with open(DIR + '/favTime.csv'.format(cv), 'rb') as fs:
        favTimeMat = pk.load(fs)
    with open(DIR + '/buyTime.csv'.format(cv), 'rb') as fs:
        buyTimeMat = pk.load(fs)
    with open(DIR + "/test_data.csv".format(cv), 'rb') as fs:
        test_data = pk.load(fs)
    with open(DIR + "/valid_data.csv".format(cv), 'rb') as fs:
        valid_data = pk.load(fs)
    with open(DIR + "/trust.csv".format(cv), 'rb') as fs:
        trustMat = pk.load(fs)
    interatctMat = ((pvTimeMat + cartTimeMat + favTimeMat + buyTimeMat) != 0) * 1
    # interatctMat = interatctMat.astype(np.bool)
    return (interatctMat, test_data, valid_data, trustMat)
    
def loadData(datasetStr, cv):
    if datasetStr == "Tianchi_time":
        return loadData2(datasetStr, cv)
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", datasetStr, 'implicit', "cv{0}".format(cv))
    log(DIR)
    with open(DIR + '/train.csv', 'rb') as fs:
        trainMat = pk.load(fs)
    with open(DIR + '/test_data.csv', 'rb') as fs:
        testData = pk.load(fs)
    with open(DIR + '/valid_data.csv', 'rb') as fs:
        validData = pk.load(fs)
    with open(DIR + '/trust.csv', 'rb') as fs:
        trustMat = pk.load(fs)
    trainMat = (trainMat!=0) * 1
    return (trainMat, testData, validData, trustMat)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def generate_sp_ont_hot(num):
    mat = sp.eye(num)
    # mat = sp.dok_matrix((num, num))
    # for i in range(num):
    #     mat[i,i] = 1
    ret = sparse_mx_to_torch_sparse_tensor(mat)
    return ret

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")

    parser.add_argument('--dataset', type=str, default='Epinions_time')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--act', type=str, default="leakyrelu")

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[8]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--test_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--reg', type=float, default=0.001,
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    
    
    
    
    # parser.add_argument('--weights_path', nargs='?', default='model/',
    #                     help='Store model path.')
    # parser.add_argument('--data_path', nargs='?', default='../Data/',
    #                     help='Input data path.')
    # parser.add_argument('--proj_path', nargs='?', default='',
    #                     help='Project path.')
    # parser.add_argument('--pretrain', type=int, default=0,
    #                     help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    # parser.add_argument('--verbose', type=int, default=1,
    #                     help='Interval of evaluation.')
    
    # parser.add_argument('--embed_size', type=int, default=16,
    #                     help='Embedding size.')
    # parser.add_argument('--layer_size', nargs='?', default='[16,16,16]',
    #                     help='Output sizes of every layer')
    
    # parser.add_argument('--regs', nargs='?', default='[1e-5]',
    #                     help='Regularizations.')
    # parser.add_argument('--model_type', nargs='?', default='ngcf',
    #                     help='Specify the name of model (ngcf).')
    # parser.add_argument('--adj_type', nargs='?', default='norm',
    #                     help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    # parser.add_argument('--alg_type', nargs='?', default='ngcf',
    #                     help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
    
    # parser.add_argument('--gpu_id', type=int, default=0,
    #                     help='0 for NAIS_prod, 1 for NAIS_concat')

    # parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
    #                     help='Output sizes of every layer')

    # parser.add_argument('--save_flag', type=int, default=1,
    #                     help='0: Disable model saver, 1: Activate model saver')

    # parser.add_argument('--test_flag', nargs='?', default='part',
    #                     help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # parser.add_argument('--report', type=int, default=0,
    #                     help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()



    