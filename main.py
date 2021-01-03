'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

#Yelp python .\main.py --reg 0.01 test_hr=0.7587, test_ndcg=0.5157

import torch as t
import torch.optim as optim

from NGCF import NGCF
from utils import parse_args
from ToolScripts.TimeLogger import log
from utils import loadData
from utils import normalize_adj
from scipy.sparse import csr_matrix
from utils import sparse_mx_to_torch_sparse_tensor
import numpy as np
import scipy.sparse as sp
from BPRData import BPRData
import torch.utils.data as dataloader
import evaluate
import warnings
import time
import pickle
import random
warnings.filterwarnings('ignore')
device_gpu = t.device("cuda")

modelUTCStr = str(int(time.time()))
isLoadModel = True
LOAD_MODEL_PATH = ""


def saveModel(model, args):
    modelName = "NGCF_" + modelUTCStr + \
        "_dataset_" + args.dataset +\
        "_cv_" + str(args.cv_num) +\
        "_rate_" + str(args.rate) +\
        "_reg_" + str(args.reg) +\
        "_lr_" + str(args.lr) +\
        "_hide_dim_" + str(args.embed_size*4) +\
        "_batch_" + str(args.batch_size)

    savePath = r'../Model/' + args.dataset + r'/' + modelName + r'.pth'
    # params = {'model': model}
    if args.save == 1:
        t.save(model, savePath)
        log("save model : %s"%(modelName))
    else:
        log("model : %s"%(modelName))


def loadModel(modelName):
    model = t.load(r'../Model/' + args.dataset + r'/' + modelName + r'.pth')
    # model = checkpoint['model']
    return model

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = t.LongTensor([coo.row, coo.col])
    v = t.from_numpy(coo.data).float()
    return t.sparse.FloatTensor(i, v, coo.shape)

def sparseTest(model, sparse_norm_adj, trainMat, testMat):
    interationSum = np.sum(trainMat != 0)
    flag = int(interationSum/3)
    user_interation = np.sum(trainMat != 0, axis=1).reshape(-1).A[0]
    sort_idx = np.argsort(user_interation)
    user_interation_sort = user_interation[sort_idx]
    
    tmp = 0
    idx = []
    for i in range(user_interation_sort.size):
        if tmp >= flag:
            tmp = 0
            idx.append(i)
            continue
        else:
            tmp += user_interation_sort[i]
    print("<{0}, <{1}, <{2}".format(user_interation_sort[idx[0]], \
                                        user_interation_sort[idx[1]], \
                                        user_interation_sort[-1]))
    print("{0}, {1}, {2}".format(idx[0], idx[1]-idx[0], userNum-idx[1]))
    splitUserIdx = [sort_idx[0:idx[0]], sort_idx[idx[0]: idx[1]], sort_idx[idx[1]:]]
    # sparseTestModel(model, sparse_norm_adj, testMat, sort_idx)
    for i in splitUserIdx:
        sparseTestModel(model, sparse_norm_adj, testMat, i)

def sparseTestModel(model, sparse_norm_adj, testMat, uid):
    test_u = np.array(uid[testMat[uid].tocoo().row])
    test_v = testMat[uid].tocoo().col
    test_r = testMat[uid].tocoo().data
    rmse, mae = test(model, sparse_norm_adj, (test_u, test_v, test_r))
    log("sparse test : user num = %d, rmse = %.4f, mae = %.4f"%(uid.size, rmse, mae))



def test(model, sparse_norm_adj, data_loader, top_k, drop_flag=False, save=False):
    HR, NDCG = [], []
    user_embed, item_embed = model(sparse_norm_adj, drop_flag=False)
    for user, item_i in data_loader:
        user = user.long().cuda()
        item_i = item_i.long().cuda()
        userEmbed = user_embed[user]
        testItemEmbed = item_embed[item_i]
        pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)
        batch = int(user.cpu().numpy().size/101)

        assert user.cpu().numpy().size % 101 ==0
        for i in range(batch):
            batch_scores = pred_i[i*101: (i+1)*101].view(-1)
            _, indices = t.topk(batch_scores, top_k)
            tmp_item_i = item_i[i*101: (i+1)*101]
            recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
            gt_item = tmp_item_i[0].item()
            HR.append(evaluate.hit(gt_item, recommends))
            NDCG.append(evaluate.ndcg(gt_item, recommends))
        if save:
            return HR, NDCG
        else:
            return np.mean(HR), np.mean(NDCG)


if __name__ == '__main__':
    np.random.seed(29)
    t.manual_seed(29)
    t.cuda.manual_seed(29)
    random.seed(29)

    args = parse_args()
    print(args)
    
    trainMat, testData, _, trustMat = loadData(args.dataset, args.cv)
    # trainMat, testMat, validMat, testData, validData = loadData(args.dataset, args.cv)
    userNum, itemNum = trainMat.shape
    train_coo = trainMat.tocoo()
    train_u, train_v = train_coo.row, train_coo.col

    train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()

    test_data = testData
    # valid_data = validData

    train_dataset = BPRData(train_data, itemNum, trainMat, 1, True)
    test_dataset = BPRData(test_data, itemNum, trainMat, 0, False)
    # valid_dataset = BPRData(valid_data, itemNum, trainMat, 0, False)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = dataloader.DataLoader(test_dataset, batch_size=101*args.test_size, shuffle=False, num_workers=0)
    # valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=101*args.test_size, shuffle=False, num_workers=0)

    userNum, itemNum = trainMat.shape

    u_i_adj = (trainMat != 0) * 1 
    i_u_adj = u_i_adj.T#.tocoo().tocsr()

    a = csr_matrix((userNum, userNum))
    b = csr_matrix((itemNum, itemNum))
    adj = sp.vstack([sp.hstack([trustMat, u_i_adj]), sp.hstack([i_u_adj, b])])
    # adj = sp.vstack([sp.hstack([a, u_i_adj]), sp.hstack([i_u_adj, b])])


    norm_adj = normalize_adj(adj + sp.eye(adj.shape[0])) 
    norm_adj = norm_adj.tocsr()
    sparse_norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj).cuda()


    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(userNum,
                 itemNum,
                 args,
                 device_gpu).to(device_gpu)

    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    cvWait = 0
    bestHR = 0
    #train
    for epoch in range(args.epoch):
        train_loader.dataset.ng_sample()
        log("start train")
        epoch_loss = 0
        for user, item_i, item_j in train_loader:

            user_idx = user.long().cuda()
            item_i_idx = item_i.long().cuda()
            item_j_idx = item_j.long().cuda()

            user_embed, item_embed = model(sparse_norm_adj, drop_flag=True)
            userEmbed = user_embed[user_idx]
            posEmbed = item_embed[item_i_idx]
            negEmbed = item_embed[item_j_idx]

            pred_i = t.sum(t.mul(userEmbed, posEmbed), dim=1)
            pred_j = t.sum(t.mul(userEmbed, negEmbed), dim=1)


            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = (bprloss + args.reg * regLoss)/args.batch_size

            # loss, mf_loss = model.create_bpr_loss(userEmbed, posEmbed, negEmbed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += bprloss.item()
        log("train epoch %d, loss = %.2f"%(epoch, epoch_loss))
        
        test_hr, test_ndcg = test(model, sparse_norm_adj, test_loader, args.top_k, drop_flag=False, save=False)
        log("test epoch %d, test_hr=%.4f, test_ndcg=%.4f\n"%(epoch, test_hr, test_ndcg))

        if test_hr > bestHR:
            bestHR = test_hr
            bestNDCG = test_ndcg
            cvWait = 0
            # saveModel(model, args)
        else:
            cvWait += 1
            log("cvWait = %d"%(cvWait))
        if cvWait == 5:
            break

    