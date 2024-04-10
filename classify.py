import pandas as pd
import numpy as np
import joblib
from utility import extract
from esm.pretrained import load_model_and_alphabet_local
import os
import torch
import esm
from sklearn.model_selection import train_test_split

def predict(in_fasta, batch_size=10, maxlen = 200, min_prob = 0.5, arg_model='path/arg_model.pkl',
            cat_model='path/cat_model.pkl',cat_index='/home/amit/PLM-ARG/models/Category_Index.csv',output_file='higarg_out.tsv'):
    arg_model = joblib.load(arg_model)#joblib.load('models/arg_model.pkl')
    cat_model = joblib.load(cat_model)#joblib.load('models/cat_model.pkl')
    cat_index = np.loadtxt(cat_index,dtype = str,delimiter = ",").tolist()#np.loadtxt('models/Category_Index.csv',dtype = str,delimiter = ",").tolist()

    if os.path.exists('path/models/seq_id.txt') and os.path.exists('path/embedding_res.txt'):
        print("Found embedding representation for each protein sequence ...")
        # Load seq_id and embedding_res from files
        seq_id = np.loadtxt('path/models/seq_id.txt', delimiter=",", dtype=str)
        embedding_res = np.loadtxt('path/models/embedding_res.txt', delimiter=",", dtype=float)
    else:
        # load ESM-1b model
        print("Loading the ESM-2 model for protein embedding ...")
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model.eval()
        try:
            model, alphabet = load_model_and_alphabet_local('path/models/esm2_t36_3B_UR50D.pt')
            model.eval()
        except IOError:
            print("The ESM-1b model is not accessible.")
    
        seq_id, embedding_res = extract(in_fasta, alphabet, model, repr_layers = [32], 
                                        batch_size = batch_size, max_len= maxlen)
    # X_train, embedding_res, Y_train, seq_id = train_test_split(embedding_res, seq_id, test_size=0.2, random_state=42)

    seq_num = len(seq_id)
    cat_num = len(cat_index)
    pred_res = pd.DataFrame({'seq_id':seq_id, 'pred':''})
    pred_res = pd.concat([pred_res, pd.DataFrame(data = np.zeros((seq_num,cat_num+1),dtype='float64'),
                     columns= ['ARG']+cat_index)], axis = 1)
    # predict ARGs
    pred_res['ARG'] = arg_model.predict_proba(embedding_res)[:,1]
    # predict Category
    arg_ind = np.where(pred_res['ARG']>min_prob)[0].tolist()
    if len(arg_ind) > 0:
        cat_out = cat_model.predict_proba(embedding_res[arg_ind,])
    for i in range(len(cat_out)):
        pred_res.iloc[arg_ind, i + 3] = cat_out[i][:, 1]

    for i in arg_ind:
        cats = [cat_index[k] for k, v in enumerate(pred_res.iloc[i, 3:]) if v >= 0.5]
        pred_res.iloc[i, 1] = ';'.join(cats)
    pred_res.to_csv(output_file, sep = '\t', index=0)
    #return pred_res

