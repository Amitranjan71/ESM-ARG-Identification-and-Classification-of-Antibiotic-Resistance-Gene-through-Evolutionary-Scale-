import torch
from esm.pretrained import load_model_and_alphabet_local
import joblib
import numpy as np
from xgboost import XGBClassifier
from utility import extract, get_label
from sklearn.multioutput import MultiOutputClassifier
import os
from sklearn.model_selection import train_test_split

def train(in_fasta, maxlen = 200, min_seq = 5, batch_size = 10,
          arg_model = 'path/models/arg_model.pkl', cat_model = 'path/models/cat_model.pkl',
          cat_index = 'path/models/Category_Index.csv'):
    
    if os.path.exists('path/models/seq_id.txt') and os.path.exists('path/models/embedding_res.txt'):
        print("Found embedding representation for each protein sequence ...")
        # Load seq_id and embedding_res from files
        seq_id = np.loadtxt('path/models/seq_id.txt', delimiter=",", dtype=str)
        embedding_res = np.loadtxt('path/models/embedding_res.txt', delimiter=",", dtype=float)
    else:
        # load ESM-1b model
        print("Loading the ESM-2 model for protein embedding ...")
        try:
            model, alphabet = load_model_and_alphabet_local('path/models/esm2_t36_3B_UR50D.pt')
            model.eval()
        except IOError:
            print("The ESM-2 model is not accessible.")
        # If files don't exist, generate the embedding vectors
        print("Generating embedding representation for each protein sequence ...")
        seq_id, embedding_res = extract(in_fasta, alphabet, model, repr_layers=[32],
                                        batch_size=batch_size, max_len=maxlen)
        # Save seq_id and embedding_res to files
        np.savetxt('path/models/seq_id.txt', seq_id, delimiter=",", fmt='%s')
        np.savetxt('path/models/embedding_res.txt', embedding_res, delimiter=",", fmt='%s')
        

    # get categories for training
    print("Get the resistance categories with more than "+ str(min_seq) + " proteins ...")
    Label_ID, ARG_Category = get_label(seq_id)
    np.savetxt(cat_index, ARG_Category, delimiter=",", fmt='%s')
    # training with XGBoost
    X = embedding_res
    Y = Label_ID
    X, p, Y, q = train_test_split(X, Y, test_size=0.1, random_state=42)
    ## 1. train model for ARG identification
    print("Training for ARG identification ...")
    model1 = XGBClassifier(learning_rate=0.1, objective='binary:logistic',
                           max_depth = 7, n_estimators = 200)
    model1.fit(X, Y[:,0])
    joblib.dump(model1, arg_model)
    print("Training for resistance category classification ...")
    arg_ind = Y[:,0] == 1
    ARG_X = X[arg_ind,:]
    ARG_Y = Y[arg_ind,1:]
    model2 = MultiOutputClassifier(XGBClassifier(learning_rate=0.1, 
                                                    objective='binary:logistic',
                                                    max_depth = 7, n_estimators = 200))
    model2.fit(ARG_X, ARG_Y)
    joblib.dump(model2, cat_model)
    
   
