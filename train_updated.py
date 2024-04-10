import torch
from esm.pretrained import load_model_and_alphabet_local
import joblib
import numpy as np
from xgboost import XGBClassifier
from utility import extract, get_label
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import roc_auc_score


from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
import numpy as np
import joblib
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import numpy as np

def train(in_fasta, maxlen = 200, min_seq = 5, batch_size = 10,
          arg_model='path/models/arg_model.pkl', cat_model='path/models/cat_model.pkl',
          cat_index='path/models/Category_Index.csv'):
        
    if os.path.exists('path/models/seq_id.txt') and os.path.exists('path/models/embedding_res.txt'):
        print("Found embedding representation for each protein sequence ...")
        # Load seq_id and embedding_res from files
        seq_id = np.loadtxt('path/models/seq_id.txt', delimiter=",", dtype=str)
        embedding_res = np.loadtxt('path/models/embedding_res.txt', delimiter=",", dtype=float)
    else:
        # load ESM-1b model
        print("Loading the ESM-2 model for protein embedding ...")
        try:
            model, alphabet = load_model_and_alphabet_local('path/esm2_t36_3B_UR50D.pt')
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
    print("Get the resistance categories with more than " + str(min_seq) + " proteins ...")
    Label_ID, ARG_Category = get_label(seq_id)
    np.savetxt(cat_index, ARG_Category, delimiter=",", fmt='%s')
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(embedding_res, Label_ID, test_size=0.2, random_state=42)


    # Define models with specified parameters and paths for saving
    models = [
    (XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42), 
     '/path/models/arg_model_XGB.pkl'),
    (RandomForestClassifier(n_estimators=250, min_samples_split=5, max_depth=7, random_state=42), 
     'path/models/arg_model_RandomForest.pkl'),
    (SVC(probability=True, kernel='linear', gamma='scale', C=0.1, random_state=42), 
     'path/models/arg_model_SVC.pkl'),
    (Lopath/models/arg_model_LogisticRegression.pkl'),
    (GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42), 
     'path/models/arg_model_GradientBoosting.pkl'),
    (KNeighborsClassifier(weights='distance', n_neighbors=11, algorithm='ball_tree'), 
     'path/models/arg_model_KNeighbors.pkl'),
    (DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=8, max_depth=7), 
     'path/models/arg_model_DecisionTree.pkl'),
    (AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=150, learning_rate=0.1), 
     'path/models/arg_model_AdaBoost.pkl'),
    (BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=150, max_samples=0.7, max_features=0.5, 
                       random_state=42), 'path/models/arg_model_BaggingClassifier.pkl'),
    (GaussianNB(), 'path/models/arg_model_GaussianNB.pkl'),
    (MLPClassifier(hidden_layer_sizes=(100, 50, 100), alpha=0.01, activation='tanh', random_state=42), 
     'path/models/arg_model_MLPClassifier.pkl'),
    (ExtraTreesClassifier(n_estimators=150, min_samples_split=12, max_depth=7, random_state=42), 
     'path/models/arg_model_ExtraTreesClassifier.pkl')
    ]

    # Loop through each model
    for idx, (model, save_path) in enumerate(models, start=1):
        print(f"Training and evaluating Model {idx} ({model.__class__.__name__})")

        # Create a 5-fold cross-validation object
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        Initialize arrays to store metrics for each fold
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_mccs = []

        # Loop through the folds
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, Y_train), start=1):
            # Split the data into training and validation sets for this fold
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            Y_fold_train, Y_fold_val = Y_train[train_index], Y_train[val_index]

            # Fit the model on the training set for this fold
            model.fit(X_fold_train, Y_fold_train[:, 0])

            # Predict on the validation set for this fold
            fold_predictions = model.predict(X_fold_val)

            # Calculate precision, recall, F1 score, and MCC for this fold
            fold_accuracy = accuracy_score(Y_fold_val[:, 0], fold_predictions)
            fold_precision = precision_score(Y_fold_val[:, 0], fold_predictions)
            fold_recall = recall_score(Y_fold_val[:, 0], fold_predictions)
            fold_f1_score = f1_score(Y_fold_val[:, 0], fold_predictions)
            fold_mcc = matthews_corrcoef(Y_fold_val[:, 0], fold_predictions)

            # Store metrics for later evaluation
            all_accuracies.append(fold_accuracy)
            all_precisions.append(fold_precision)
            all_recalls.append(fold_recall)
            all_f1_scores.append(fold_f1_score)
            all_mccs.append(fold_mcc)

            # Print metrics for this fold
            print(f"  Fold {fold}: Accuracy={fold_accuracy:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}, "
                  f"F1-score={fold_f1_score:.4f}, MCC={fold_mcc:.4f}")

        # Calculate overall metrics using metrics from all folds
        overall_accuracy = np.mean(all_accuracies)
        overall_precision = np.mean(all_precisions)
        overall_recall = np.mean(all_recalls)
        overall_f1_score = np.mean(all_f1_scores)
        overall_mcc = np.mean(all_mccs)

        print(f"  Overall Accuracy: {overall_accuracy:.4f}")
        print(f"  Overall Precision: {overall_precision:.4f}")
        print(f"  Overall Recall: {overall_recall:.4f}")
        print(f"  Overall F1-score: {overall_f1_score:.4f}")
        print(f"  Overall MCC: {overall_mcc:.4f}")

        # Evaluate on the test set
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(Y_test[:, 0], test_pred)
        test_precision = precision_score(Y_test[:, 0], test_pred)
        test_recall = recall_score(Y_test[:, 0], test_pred)
        test_f1_score = f1_score(Y_test[:, 0], test_pred)
        test_mcc = matthews_corrcoef(Y_test[:, 0], test_pred)

        print(f"Test Metrics for Model {idx} ({model.__class__.__name__}): "
              f"Accuracy={test_acc:.4f}, Precision={test_precision:.4f}, "
              f"Recall={test_recall:.4f}, F1-score={test_f1_score:.4f}, MCC={test_mcc:.4f}")

        # Save the trained model
        joblib.dump(model, save_path)
        print(f"  Model saved to: {save_path}")

    
 
#     ## Hyperparameter tuning for model2

    X_train, X_test, Y_train, Y_test = train_test_split(embedding_res, Label_ID, test_size=0.1, random_state=42)
    
    print("Training for resistance category classification ...")
    arg_ind = Y_train[:, 0] == 1
    ARG_X = X_train[arg_ind, :]
    ARG_Y = Y_train[arg_ind, 1:]
    cat_model = 'path/models/arg_model_classifier_XGB.pkl'

    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_mccs = []

    # Loop through the folds
    for fold, (train_index, val_index) in enumerate(kf.split(ARG_X, ARG_Y), start=1):
        # Split the data into training and validation sets for this fold
        X_fold_train, X_fold_val = ARG_X[train_index], ARG_X[val_index]
        Y_fold_train, Y_fold_val = ARG_Y[train_index], ARG_Y[val_index]

        # Define and train the model for this fold
        model_classifier_XGB = MultiOutputClassifier(
            XGBClassifier(learning_rate=0.1, objective='binary:logistic', max_depth=10, n_estimators=500)
        )
        model_classifier_XGB.fit(X_fold_train, Y_fold_train)

        # Predict on the validation set for this fold
        fold_predictions = model_classifier_XGB.predict(X_fold_val)

        # Calculate metrics for this fold
        fold_accuracy = accuracy_score(Y_fold_val, fold_predictions)
        fold_precision = precision_score(Y_fold_val, fold_predictions, average='micro')
        fold_recall = recall_score(Y_fold_val, fold_predictions, average='micro')
        fold_f1_score = f1_score(Y_fold_val, fold_predictions, average='micro')
        
        cm = multilabel_confusion_matrix(Y_fold_val[:, 0], fold_predictions)
        tn, fp, fn, tp = cm.ravel()
        fold_mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0

        # Store metrics for later evaluation
        all_accuracies.append(fold_accuracy)
        all_precisions.append(fold_precision)
        all_recalls.append(fold_recall)
        all_f1_scores.append(fold_f1_score)
        all_mccs.append(fold_mcc)

        # Print metrics for this fold
        print(f"Fold {fold}: Accuracy={fold_accuracy:.4f}, Precision={fold_precision:.4f}, "
              f"Recall={fold_recall:.4f}, F1-score={fold_f1_score:.4f}, MCC={fold_mcc:.4f}")

    # Calculate overall metrics using metrics from all folds
    overall_accuracy = np.mean(all_accuracies)
    overall_precision = np.mean(all_precisions)
    overall_recall = np.mean(all_recalls)
    overall_f1_score = np.mean(all_f1_scores)
    overall_mcc = np.mean(all_mccs)

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1-score: {overall_f1_score:.4f}")
    print(f"Overall MCC: {overall_mcc:.4f}")
    
    # Evaluate on the test set
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(Y_test[:, 0], test_pred)
    test_precision = precision_score(Y_test[:, 0], test_pred)
    test_recall = recall_score(Y_test[:, 0], test_pred)
    test_f1_score = f1_score(Y_test[:, 0], test_pred)
    cm = multilabel_confusion_matrix(Y_test[:, 0], test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    #test_mcc = matthews_corrcoef(Y_test[:, 0], test_pred)

    print(f"Test Metrics for Model {idx} ({model.__class__.__name__}): "
          f"Accuracy={test_acc:.4f}, Precision={test_precision:.4f}, "
          f"Recall={test_recall:.4f}, F1-score={test_f1_score:.4f}, MCC={test_mcc:.4f}")

    # Save the trained model
    joblib.dump(model_classifier_XGB, cat_model)
    print(f"Model saved to: {cat_model}")

  
    # Compute ROC curve and ROC area for Model1
#     fpr1, tpr1, _ = roc_curve(Y_test[:, 0], test_pred1)
#     roc_auc1 = auc(fpr1, tpr1)

#     fpr2, tpr2, _ = roc_curve(Y_test[:, 0], test_pred2)
#     roc_auc2 = auc(fpr2, tpr2)

#     fpr3, tpr3, _ = roc_curve(Y_test[:, 0], test_pred3)
#     roc_auc3 = auc(fpr3, tpr3)

#     fpr4, tpr4, _ = roc_curve(Y_test[:, 0], test_pred4)
#     roc_auc4 = auc(fpr4, tpr4)

#     fpr5, tpr5, _ = roc_curve(Y_test[:, 0], test_pred5)
#     roc_auc5 = auc(fpr5, tpr5)

#     fpr6, tpr6, _ = roc_curve(Y_test[:, 0], test_pred6)
#     roc_auc6 = auc(fpr6, tpr6)

#     fpr7, tpr7, _ = roc_curve(Y_test[:, 0], test_pred7)
#     roc_auc7 = auc(fpr7, tpr7)

#     fpr8, tpr8, _ = roc_curve(Y_test[:, 0], test_pred8)
#     roc_auc8 = auc(fpr8, tpr8)

#     fpr9, tpr9, _ = roc_curve(Y_test[:, 0], test_pred9)
#     roc_auc9 = auc(fpr9, tpr9)

#     fpr10, tpr10, _ = roc_curve(Y_test[:, 0], test_pred10)
#     roc_auc10 = auc(fpr10, tpr10)

#     fpr11, tpr11, _ = roc_curve(Y_test[:, 0], test_pred11)
#     roc_auc11 = auc(fpr11, tpr11)

#     fpr12, tpr12, _ = roc_curve(Y_test[:, 0], test_pred12)
#     roc_auc12 = auc(fpr12, tpr12)


    # Save ROC curve for Model1
#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='XGB (area = {:.2f})'.format(roc_auc1))
#     plt.plot(fpr2, tpr2, color='blue', lw=2, label='RF (area = {:.2f})'.format(roc_auc2))
#     plt.plot(fpr3, tpr3, color='red', lw=2, label='SVC (area = {:.2f})'.format(roc_auc3))
#     plt.plot(fpr4, tpr4, color='pink', lw=2, label='LR (area = {:.2f})'.format(roc_auc4))
#     plt.plot(fpr5, tpr5, color='green', lw=2, label='GradientBoosting (area = {:.2f})'.format(roc_auc5))
#     plt.plot(fpr6, tpr6, color='navy', lw=2, label='KNeighbors (area = {:.2f})'.format(roc_auc6))
#     plt.plot(fpr7, tpr7, color='purple', lw=2, label='DecisionTree (area = {:.2f})'.format(roc_auc7))
#     plt.plot(fpr8, tpr8, color='orange', lw=2, label='AdaBoost (area = {:.2f})'.format(roc_auc8))
#     plt.plot(fpr9, tpr9, color='brown', lw=2, label='Bagging (area = {:.2f})'.format(roc_auc9))
#     plt.plot(fpr10, tpr10, color='cyan', lw=2, label='GaussianNB (area = {:.2f})'.format(roc_auc10))
#     plt.plot(fpr11, tpr11, color='magenta', lw=2, label='MLP (area = {:.2f})'.format(roc_auc11))
#     plt.plot(fpr12, tpr12, color='olive', lw=2, label='ExtraTrees (area = {:.2f})'.format(roc_auc12))


#     #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve for ARG identification models')
#     plt.legend(loc='lower right')
#     plt.savefig('path/models/roc_curve_ARG_identification.png')
    
