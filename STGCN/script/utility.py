import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
import os
import pandas as pd
from sklearn import preprocessing
def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')
    adj=dir_adj
    # print('\n这是adj',type(adj))
    # Symmetrizing an adjacency matrix
    # adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler,train_iter,path):
    #train_iter原始训练数据
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        
        i=1#保存y 和 y_pred 的batch 号码
        
        # path = f"./data/tanggu/result/predict_result{folder}"
        # try:
        #     os.makedirs(path)
        # except:
        #     print('test已存在该文件夹')
        x_num=33
        n_vertex = train_iter.shape[1]
        y_num=n_vertex-x_num
        each_mae=np.empty((1,y_num))
        each_maer=np.empty((1,y_num))
        for x, y in data_iter:
            
            #得到y_train的fit_transform
            y_data=train_iter.iloc[:,x_num:]
            zscore = preprocessing.StandardScaler()
            y_data= zscore.fit_transform(y_data)
            # print('\n均值为:',zscore.mean_,'标准差为',zscore.scale_)
            # 逆转化回来
            y_save=zscore.inverse_transform(y.cpu().numpy())#原始的该批次的y 有_save的是*列数的
            # x_data[head: tail] = x.reshape(288, 24, 33)
            np.savetxt(path+f'/y_save{i}.csv',y_save,delimiter=',')
            np.savetxt(path+f'/y_{i}.csv',y.cpu().numpy(),delimiter=',')
          
            # print(type(zscore.inverse_transform(y.cpu().numpy())))
            y = y_save.reshape(-1)
            
            #再次得到y_train的fit_transform，注意不是scaler.而是zscore,估计保存到这个类里了
            y_data=train_iter.iloc[:,x_num:]
            zscore = preprocessing.StandardScaler()
            y_data= zscore.fit_transform(y_data)
            
            #逆转化回来
            y_pred_save=zscore.inverse_transform(model(x).view(len(x), -1).cpu().numpy())
            np.savetxt(path+f'/y_pred_save{i}.csv',y_pred_save,delimiter=',')
            y_pred = y_pred_save.reshape(-1)
            
            each_d=np.abs(y_pred_save-y_save)#得到每个y的绝对差
            e_r=each_d/y_save
            
            # print(each_d)
            # print(y_save)
            # print(e_r)
            each_mae=np.concatenate((each_mae,each_d),axis=0)
            each_maer=np.concatenate((each_maer,e_r),axis=0)
            # each_mean=np.mean(each_mae,axis=0)#test所有的差值
            
            #原有的代码
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
            i=i+1
       
        each_mae=each_mae[1:]
        each_maer=each_maer[1:]
        np.savetxt(path+'each_maer.csv',each_maer,delimiter=',')
        each_mean=np.mean(each_mae,axis=0)#test所有的差值
        each_r_mean=np.mean(each_maer,axis=0)#test所有的差值
        MAE = np.array(mae).mean()
    
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))
        reslt_dict= {'MAE':MAE,'RMSE': RMSE,'WMAPE':WMAPE,'each_mean':each_mean,'each_r_mean':each_r_mean}#存起来
        reslt_dict = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in reslt_dict.items()]))
        reslt_dict.to_csv(path+r"\reslt_test.csv")
        print(reslt_dict)
        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE
