import sys
import random
import pickle
import traceback
import pandas as pd
import hyperopt as hp
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, SparkTrials
from hyperopt.early_stop import no_progress_loss

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight


def f1_loss(y_true, y_pred):
    #  y_true = dtrain.get_label()
    err = fbeta_score(y_true, np.rint(y_pred), beta=1)
    return -(np.log(err, out=np.array(-10.0), where=(err!=0)))

def loss_tuning(score):
    return -(np.log(score, out=np.array(-10.0), where=(score!=0)))


features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


def objective(space) -> dict:
    
    space['max_depth'] = int(space['max_depth'])
    #space['max_delta_step'] = int(space['max_delta_step'])
    space['colsample_bytree'] = int(space['colsample_bytree'])
    space['max_bin'] = int(space['max_bin'])
    
    train_x, valid_x, train_y, valid_y = train_test_split(tune_x, tune_y, test_size=0.20, stratify=tune_y)
    
    evaluation = [(valid_x, valid_y)]
    
    weight = compute_sample_weight(class_weight='balanced', y=train_y)
    
    #quantile matrix
    #dtrain = xgboost.QuantileDMatrix(max_bin=space['max_bin'], data=tune_x, label=tune_y, weight=weight, )
    # dval = xgboost.QuantileDMatrix(data=valid_x, label=valid_y)
    
    clf=XGBClassifier(**space)
    
    clf.fit(train_x, train_y,
            eval_set = evaluation,
            verbose = False,
            sample_weight=weight)
    
    # train score
    pred_labels_train = np.rint(clf.predict(train_x)) # gli score sono approssimati con la soglia a 0.5
    f1_train = fbeta_score(train_y, pred_labels_train, beta=1, zero_division=0) # fb score con b=1 = f1 score
    
    # loss score tuning
    pred_labels = np.rint(clf.predict(valid_x)) # gli score sono approssimati con la soglia a 0.5
    f1 = fbeta_score(valid_y, pred_labels, beta=1, zero_division=0)
    score = loss_tuning(f1)
    
    print(f"best boosting iteration: {clf.best_iteration}")
    #print(f"used parameters : {clf.get_params(deep=False)}")
    print(f"train score:{f1_train} \nvalidation score:{f1}")
    
    precision = precision_score(pred_labels, valid_y, zero_division=0)
    recall =recall_score(pred_labels, valid_y, zero_division=0)
    
    return {'status': STATUS_OK,
            'loss': score,
            # 'loss_variance': scor_var, # per la cross validation
            'f1_train':f1_train,
            'f1_val':f1,
            'precision': precision,
            'recall': recall,
            }

if __name__ == "__main__":
        
    #data = pd.read_parquet('../dati/dataset_processed.parquet')
    data = pd.read_csv('data/creditcard.csv')
    
    tune_x, test_x, tune_y, test_y = train_test_split(data[features], data['Class'], test_size=0.20, stratify=data['Class']) # final test split
    
    # new_df = tune_x
    # #balaning the dataset
    # new_df['Class']=tune_y
    # new_df = new_df.sample(frac=1)
    # fraud_df = new_df.loc[new_df['Class'] == 1]
    # non_fraud_df = new_df.loc[new_df['Class'] == 0][:fraud_df.Class.sum()]
    # normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    # # Shuffle dataframe rows
    # balanced_df = normal_distributed_df.sample(frac=1)

    random_seed = random.randint(0, 100000) #seed buoni: 
    #random_seed = 
    
    prop_1_low=int(np.log(tune_y.sum()/len(tune_y)))
    prop_1_up=int(np.log(tune_y.sum()/len(tune_y)*100))
    
    print(f"proportion of class_1: {tune_y.sum()/len(tune_y)}")

    space={'max_depth': hp.quniform("max_depth", 4, 14, 2),
           'eta' : hp.loguniform("eta", -4 ,-1),
           'max_bin' : hp.quniform("max_bin", 10, 100, 2), #sembra aumentare la recall
           'gamma': hp.uniform("gamma", 0 ,30), # riduzione di loss per un ulteriore split
           #'max_delta_step' : hp.uniform('max_delta_step', 10, 50), #  it might help in logistic regression when class is extremely imbalanced
           #'scale_pos_weight' : hp.loguniform('scale_pos_weight', -10, 0),
           'reg_alpha' : hp.uniform('reg_alpha', 10, 100),
           #'reg_lambda' : hp.uniform('reg_lambda', 0, 50),
           'colsample_bytree' : hp.quniform('colsample_bytree', 0.7, 1, 0.1),
           'subsample' : hp.loguniform('subsample', prop_1_low, prop_1_up),
           'n_estimators' : 1000,

           # 'subsample' : 1,
           'early_stopping_rounds' : 100,
           #'sampling_method' : 'gradient_based',
           'device' : 'gpu', # o cpu, cuda. manca di capire come usare la gpu con queste configurazioni
           'eval_metric' : f1_loss, # f1 solo per device cpu. Attention!! se custom metric allora xgboost di default cercherà di minimizzarla
           'tree_method' : 'hist', # exact solo per cpu
           'objective' : 'binary:logistic',
           'seed': random_seed
        }
    
    traials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 9000, # il range calcolato va da 7200 a 14400 per il feature space completo
                            early_stop_fn=no_progress_loss(100), # %1 max eval
                            trials = traials,
                            rstate=np.random.default_rng(random_seed),
                           ) # usare spark trials con core = max_eval/10. l'istanza di calcolo ne ha 4
        
    print(f"used seed: {random_seed}")
    pickle.dump(traials, open("results/xgb_traials.pkl", "wb"))

    parameters = ['max_depth','eta','max_bin','reg_alpha','colsample_bytree','subsample','gamma'] #'reg_lambda',
    metrics = ['recall','precision','f1_val','f1_train']
    results = pd.DataFrame()
    
    for x in range(0,len(parameters)):
        values = [traials.trials[i]['misc']['vals'][parameters[x]][0] for i in range(0, len(traials.trials))]
        results[parameters[x]] = values
        results['recall'] = [traials.trials[i]['result']['recall'] for i in range(0, len(traials.trials))]
        results['precision'] = [traials.trials[i]['result']['precision'] for i in range(0, len(traials.trials))]
        results['f1_val'] = [traials.trials[i]['result']['f1_val'] for i in range(0, len(traials.trials))]
        results['f1_train'] = [traials.trials[i]['result']['f1_train'] for i in range(0, len(traials.trials))]
    results['loss'] = traials.losses()
    
    results.to_csv('results/xgboostTuning_results.csv', index=False)

    best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
    #best_hyperparams['max_delta_step'] = int(best_hyperparams['max_delta_step'])
    best_hyperparams['colsample_bytree'] = int(best_hyperparams['colsample_bytree'])
    best_hyperparams['max_bin'] = int(best_hyperparams['max_bin'])
    
    print(f"Number of finished trials: {traials.trials[-1]['tid']+1}")
    print(f"The best hyperparameters are : {best_hyperparams}")
    
    try:
        weight = compute_sample_weight(class_weight='balanced', y=tune_y)
        final_model = XGBClassifier(**best_hyperparams)
        scores = final_model.fit(tune_x, tune_y, sample_weight=weight, verbose = False).predict_proba(test_x)
        pred_labels = final_model.predict(test_x)
                                    
        xgboost_conf = confusion_matrix(test_y, pred_labels)
        
        print(f"\n Confusion matrix on test set:")

        plt.figure(figsize=(8, 6))
        sns.heatmap(xgboost_conf, annot=True, cmap=plt.cm.copper)
        plt.title("Xgboost Classifier \n Confusion Matrix", fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.show()

        precision, recall, thresholds = precision_recall_curve(test_y, scores[:,1])

        threshold = np.append(thresholds,np.nan) # threshold are #precsion_scores-1

        data_treshold = pd.DataFrame()
        data_treshold['precision'] = precision
        data_treshold['recall'] = recall
        data_treshold['thresholds'] = threshold

        data_treshold.to_csv('results/xgboostTuning_data_treshold.csv', index=False)
        
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info() 
        tb = traceback.TracebackException(exc_type, exc_value, exc_tb) 
        print(''.join(tb.format_exception_only())) 
        print(f"error \n {traceback.print_exc()} ")