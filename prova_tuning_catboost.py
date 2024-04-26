import sys
import random
import pickle
import traceback
import pandas as pd
import catboost as cb
import hyperopt as hp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, SparkTrials
from hyperopt.early_stop import no_progress_loss

from catboost import Pool
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

# per creare una metrica di validation custom
# class UserDefinedMetric(object):
#     def is_max_optimal(self):
#         # Returns whether great values of metric are better
#         pass

#     def evaluate(self, approxes, target, weight):
#         # approxes is a list of indexed containers
#         # (containers with only __len__ and __getitem__ defined),
#         # one container per approx dimension.
#         # Each container contains floats.
#         # weight is a one dimensional indexed container.
#         # target is a one dimensional indexed container.

#         # weight parameter can be None.
#         # Returns pair (error, weights sum)
#         pass

#     def get_final_error(self, error, weight):
#         # Returns final value of metric based on error and weight
#         pass


def loss_tuning(score):
    return -(np.log(score, out=np.array(-10.0), where=(score!=0)))


features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'] #'timeFromLastTransaction'

def objective(space) -> dict:

    train_x, valid_x, train_y, valid_y = train_test_split(tune_x, tune_y, test_size=0.20, stratify=tune_y)
    
    weight = compute_sample_weight(class_weight='balanced', y=train_y)

    clf = cb.CatBoostClassifier(**space)

    # pruning_callback = CatBoostPruningCallback(trial, "F1")
    
    quantized_dataset_path = 'data/quantized_dataset.bin'

    # save quantized dataset
    train_dataset = Pool(train_x, train_y, weight=weight,)
    train_dataset.quantize()
    train_dataset.save(quantized_dataset_path)

    # fit multiple models w/o dataset quantization
    quantized_train_dataset = Pool(data='quantized://' + quantized_dataset_path)
    
    clf.fit(
        quantized_train_dataset,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,)
        #cat_features=cat,
        #sample_weight=weight,)


   # train score
    pred_labels_train = np.rint(clf.predict(train_x)) # gli score sono approssimati con la soglia a 0.5
    f1_train = fbeta_score(train_y, pred_labels_train, beta=1, zero_division=0) # fb score con b=1 = f1 score
    
    # loss score tuning
    pred_labels = np.rint(clf.predict(valid_x)) # gli score sono approssimati con la soglia a 0.5
    f1 = fbeta_score(valid_y, pred_labels, beta=1, zero_division=0)
    score = loss_tuning(f1)
    
    print(f"best boosting iteration: {clf.get_best_iteration()}")
    print(f"train score:{f1_train} \nvalidation score:{f1}")
    print(f"used parameters : {clf.get_params()}")
    
    precision = precision_score(pred_labels, valid_y, zero_division=0)
    recall =recall_score(pred_labels, valid_y, zero_division=0)
    
    return {'status': STATUS_OK,
            'loss': score,
            'f1_train':f1_train,
            'f1_val':f1,
            'precision': precision,
            'recall': recall,
            #'loss_variance': np.var(score, ddof=1), # per la cross validation
            }


if __name__ == "__main__":
     #  cv = 2 # numero di partizioni del dataset per lo scoring
        
    data =pd.read_csv('data/creditcard.csv')
    
    tune_x, test_x, tune_y, test_y = train_test_split(data[features], data['Class'], test_size=0.20, stratify=data['Class']) # final test split
    
    random_seed = random.randint(0, 100000) #seed buoni: 
    
    prop_1_low=int(np.log(tune_y.sum()/len(tune_y)))
    prop_1_up=int(np.log(tune_y.sum()/len(tune_y)*100))

    space={
        #"boosting_type": hp.choice("boosting_type", ["Ordered", "Plain"]), #
        "grow_policy" : hp.choice("grow_policy", ["Depthwise", "Lossguide", "SymmetricTree"]),
        "depth": hp.quniform("depth", 3, 12,1),
        'learning_rate' : hp.loguniform("learning_rate", -8 , -1),
        'l2_leaf_reg': hp.uniform("l2_leaf_reg", 0, 50), # riduzione di loss per un ulteriore split
        #'scale_pos_weight' : hp.uniform('scale_pos_weight', 5e-5, 1,), #  it might help in logistic regression when class is extremely imbalanced
        "border_count" : hp.quniform("border_count", 10, 100, 2),
        'subsample' : hp.loguniform('subsample', prop_1_low, prop_1_up),
        
        "boosting_type":"Plain",
        "bootstrap_type" : "Bernoulli", # la probabilità è data da subsample
        # "bagging_temperature" : 1, #default, i weughts sono assegnati al momento del train
        "objective" : "Logloss",
        "task_type" : "GPU",
        "iterations" : 1000,
        "eval_metric": "F1",
    }
    
    # if space["boosting_type"] == "Plain":
    #     space["grow_policy"] = hp.choice("grow_policy", ["Depthwise", "Lossguide", "SymmetricTree"])
    # else:
    #     space["grow_policy"] = "SymmetricTree"
    
    traials = Trials()
    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 9000, # il range calcolato va da 7200 a 14400
                            early_stop_fn=no_progress_loss(100),
                            trials = traials,
                            rstate=np.random.default_rng(random_seed),
                           ) # usare spark trials con core = max_eval/10. l'istanza di calcolo ne ha 4
    
    print(f"used seed: {random_seed}")
    pickle.dump(traials, open("results_catboost/cb_traials.pkl", "wb"))

    parameters = ['depth', 'learning_rate', 'l2_leaf_reg', 'border_count', 'subsample', 'grow_policy']
    results = pd.DataFrame()
    for x in range(0,len(parameters)):
        values = [traials.trials[i]['misc']['vals'][parameters[x]][0] for i in range(0, len(traials.trials))]
        results[parameters[x]] = values
        results['recall'] = [traials.trials[i]['result']['recall'] for i in range(0, len(traials.trials))]
        results['precison'] = [traials.trials[i]['result']['precision'] for i in range(0, len(traials.trials))]
        results['f1_val'] = [traials.trials[i]['result']['f1_val'] for i in range(0, len(traials.trials))]
        results['f1_train'] = [traials.trials[i]['result']['f1_train'] for i in range(0, len(traials.trials))]
    results['loss'] = traials.losses()
    
    results.to_csv('results_catboost/catboostTuning_results.csv', index=False)
    
    try:
        weight = compute_sample_weight(class_weight='balanced', y=tune_y)
        final_model = cb.CatBoostClassifier(**best_hyperparams)
        scores = final_model.fit(tune_x, tune_y,
                                 verbose=False,
                                 early_stopping_rounds=100,
                                 sample_weight=weight).predict_proba(test_x)
        pred_labels = final_model.predict(test_x)
                                    
        catboost_conf = confusion_matrix(test_y, pred_labels)
        
        print(f"Confusion matrix on test set: \n")

        plt.figure(figsize=(8, 6))
        sns.heatmap(catboost_conf, annot=True, cmap=plt.cm.copper)
        plt.title("catboost Classifier \n Confusion Matrix", fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.show()

        precision, recall, thresholds = precision_recall_curve(test_y, scores[:,1])

        threshold = np.append(thresholds,np.nan) # threshold are #precsion_scores-1

        data_treshold = pd.DataFrame()
        data_treshold['precision'] = precision
        data_treshold['recall'] = recall
        data_treshold['thresholds'] = threshold
        data_treshold.to_csv('results_catboost/catboostTuning_data_treshold.csv', index=False)
        
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info() 
        tb = traceback.TracebackException(exc_type, exc_value, exc_tb) 
        print(''.join(tb.format_exception_only())) 
        print(f"error \n {traceback.print_exc()} ")

    print(f"Number of finished trials: {traials.trials[-1]['tid']+1}")
    print(f"The best hyperparameters of tuning are : {best_hyperparams}")