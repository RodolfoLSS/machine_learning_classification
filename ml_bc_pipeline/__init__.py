import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auprc, \
    grid_search_NN, grid_search_SVM, grid_search_KNN, grid_search_DT, grid_search_RF, \
    grid_search_NB
from ml_bc_pipeline.utils import BalanceDataset
import matplotlib.pyplot as plt

def main():
########################################################################################################################
##### LOAD DATA
    # Choose working directory (path).
    os.chdir("/Users/RodolfoSaldanha/Desktop/ML/project1") # Change this!
    # Name of the .xlsx file with the data.
    file_path = os.getcwd()+"/ml_project1_data.xlsx"
    # Load the data through the class Dataset.
    ds = Dataset(file_path) # from data_leader.py



########################################################################################################################
##### SPLIT IN TRAIN AND UNSEEN
    # Set seed.
    seed = 12345
    DF_train, DF_unseen = train_test_split(ds.rm_df.copy(), test_size=0.2, stratify=ds.rm_df["Response"], random_state=seed)



########################################################################################################################
##### PREPROCESS
    # The preprocess and feature engineering implemented here is based on information gathered on Feature Exploration
    # Jupyter Notebook.
    pr = Processor(DF_train, DF_unseen) # from data_preprocessing.py

########################################################################################################################
##### FEATURE ENGINEERING
    fe = FeatureEngineer(pr.training, pr.unseen)  # from feature_engineering.py

    # Apply Box-Cox Transformations and gather the best ones for each feature.
    cat = ['Education', 'Marital_Status','NmbAccCmps','Response','Age_d', 'Income_d']
    num_features = fe.training._get_numeric_data().drop(cat, axis=1).columns
    fe.box_cox_transformations(num_features, target="Response")

    # Rank input features according to Chi-Squared and Decision Tree Algorithm
    continuous_flist = fe.box_cox_features
    categorical_flist = ["Kidhome", "Teenhome", "AcceptedCmp1","AcceptedCmp2", "AcceptedCmp3",
                         "AcceptedCmp4", "AcceptedCmp5", "Complain","HasOffspring", 'DT_Acc_1',
                         'DT_MS_Single', 'DT_MS_Widow', 'DT_MS_Divorced', 'DT_E_Phd', 'DT_E_Master',
                         "DT_Age_4", "DT_Age_3", "DT_Age_2", "DT_Age_1", "DT_Income_3", "DT_Income_2", "DT_Income_1"]

    # Remove BxCxT_Recency because it's -0.9 correlated to BxCxT_RFM, which was originated from BxCxT_Recency
    continuous_flist.remove('BxCxT_Recency')

    # Chi-Square Rank
    fe.rank_features_chi_square(continuous_flist, categorical_flist)
    # DTA Rank
    fe.calc_dta_feat_worth(continuous_flist + categorical_flist, 5, 100, 10, seed)
    print("Ranked input features:\n", fe._rank)
    #print(len(continuous_flist + categorical_flist))

    # Feature Selection
    # SET selection criteria and number of top features:
    # criteria = ["chisq" or "dta]
    # n_top = [from 1 to inf+]
    # n_top = 44

    DF_train_top_chi, DF_unseen_chi = fe.get_top(criteria="chisq", n_top=47)
    DF_train_top_dta, DF_unseen_dta = fe.get_top(criteria="dta", n_top=15)

    #corr = pd.DataFrame(DF_train_top_chi.corr())
    #corr.to_csv('bla.csv')
    #corr.to_excel('bla.xls')

########################################################################################################################
##### MODEL - MLP
    """
    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}

    ##### Chisq Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

    mlp_gscv_chi = grid_search_MLP(DF_train_top, mlp_param_grid, seed)
    auprc_mlp_chi = assess_generalization_auprc(mlp_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    mlp_gscv_dta = grid_search_MLP(DF_train_top, mlp_param_grid, seed)
    auprc_mlp_dta = assess_generalization_auprc(mlp_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")
    # Can use above code to save results from model

    ##### Printing Results

    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", mlp_gscv_dta.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_mlp_chi))

    print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", mlp_gscv_chi.best_params_)
    print("DTA Selection AUPRC: {:.2f}".format(auprc_mlp_dta))

    ##### Plotting Results
    plt.show()"""

########################################################################################################################
##### MODEL - NN

    nn_param_grid = {'nn__batch_size': [25, 32, 40],
                     'nn__epochs': [50, 100, 200]}

    ##### Chisq Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi
    
    nn_gscv_chi = grid_search_NN(DF_train_top, nn_param_grid, seed)
    auprc_nn_chi = assess_generalization_auprc(nn_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection
    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    #nn_gscv_dta = grid_search_NN(DF_train_top, nn_param_grid, seed)
    #auprc_nn_dta = assess_generalization_auprc(nn_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")
    
########################################################################################################################
    
    ##### Printing Results
    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", nn_gscv_chi.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_nn_chi))

    #print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", nn_gscv_dta.best_params_)
    #print("DTA Selection AUPRC: {:.2f}".format(auprc_nn_dta))

    ##### Plotting Results
    plt.show()
    


########################################################################################################################
##### MODEL - SVM
    """
    svm_param_grid = {'svm__C': [6,7,8,9,10,11,12],
                        'svm__kernel': ['linear','rbf']}

    ##### Chisq Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

    svm_gscv_chi = grid_search_SVM(DF_train_top, svm_param_grid, seed)
    auprc_svm_chi = assess_generalization_auprc(svm_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    svm_gscv_dta = grid_search_SVM(DF_train_top, svm_param_grid, seed)
    auprc_svm_dta = assess_generalization_auprc(svm_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

########################################################################################################################

    ##### Printing Results
    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", svm_gscv_chi.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_svm_chi))

    print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", svm_gscv_dta.best_params_)
    print("DTA Selection AUPRC: {:.2f}".format(auprc_svm_dta))

    ##### Plotting Results
    plt.show()


########################################################################################################################
##### MODEL - KNN
    
    knn_param_grid = {'knn__n_neighbors':[5,6,7,8,9,10],
                    'knn__leaf_size':[1,2,3,5],
                    'knn__weights':['uniform', 'distance'],
                    'knn__algorithm':['auto', 'ball_tree','kd_tree','brute'],
                    'knn__n_jobs':[-1]}

    ##### Chisq Feature Selection


    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

    knn_gscv_chi = grid_search_KNN(DF_train_top, knn_param_grid, seed)
    auprc_knn_chi = assess_generalization_auprc(knn_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    knn_gscv_dta = grid_search_KNN(DF_train_top, knn_param_grid, seed)
    auprc_knn_dta = assess_generalization_auprc(knn_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

########################################################################################################################

    ##### Printing Results
    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", knn_gscv_chi.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_knn_chi))

    print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", knn_gscv_dta.best_params_)
    print("DTA Selection AUPRC: {:.2f}".format(auprc_knn_dta))

    ##### Plotting Results
    plt.show()

########################################################################################################################
##### MODEL - DT
    dt_param_grid = {'dt__max_features': ['auto', 'sqrt', 'log2'],
                      'dt__min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                      'dt__min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}

    ##### Chisq Feature Selection


    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

    dt_gscv_chi = grid_search_DT(DF_train_top, dt_param_grid, seed)
    auprc_dt_chi = assess_generalization_auprc(dt_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    dt_gscv_dta = grid_search_DT(DF_train_top, dt_param_grid, seed)
    auprc_dt_dta = assess_generalization_auprc(dt_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

########################################################################################################################

    ##### Printing Results
    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", dt_gscv_chi.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_dt_chi))

    print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", dt_gscv_dta.best_params_)
    print("DTA Selection AUPRC: {:.2f}".format(auprc_dt_dta))

    ##### Plotting Results
    plt.show()

########################################################################################################################
##### MODEL - RF
    rf_param_grid = {'rf__criterion':['gini','entropy'],
                    'rf__n_estimators':[10,15,20,25,30],
                    'rf__min_samples_leaf':[1,2,3],
                    'rf__min_samples_split':[3,4,5,6,7],
                    'rf__random_state':[123],
                    'rf__n_jobs':[-1]}

    ##### Chisq Feature Selection


    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

    rf_gscv_chi = grid_search_RF(DF_train_top, rf_param_grid, seed)
    auprc_rf_chi = assess_generalization_auprc(rf_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    rf_gscv_dta = grid_search_RF(DF_train_top, rf_param_grid, seed)
    auprc_rf_dta = assess_generalization_auprc(rf_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

########################################################################################################################

    ##### Printing Results
    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", rf_gscv_chi.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_rf_chi))

    print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", rf_gscv_dta.best_params_)
    print("DTA Selection AUPRC: {:.2f}".format(auprc_rf_dta))

    ##### Plotting Results
    plt.show()

########################################################################################################################
##### MODEL - NB
    nb_param_grid = {}

    ##### Chisq Feature Selection


    DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

    nb_gscv_chi = grid_search_NB(DF_train_top, nb_param_grid, seed)
    auprc_nb_chi = assess_generalization_auprc(nb_gscv_chi.best_estimator_, DF_unseen_top)
    ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

    ##### DTA Feature Selection

    DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

    nb_gscv_dta = grid_search_RF(DF_train_top, nb_param_grid, seed)
    auprc_nb_dta = assess_generalization_auprc(nb_gscv_dta.best_estimator_, DF_unseen_top)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

########################################################################################################################

    ##### Printing Results
    print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", nb_gscv_chi.best_params_)
    print("Chisq Selection AUPRC: {:.2f}".format(auprc_nb_chi))

    print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", nb_gscv_dta.best_params_)
    print("DTA Selection AUPRC: {:.2f}".format(auprc_nb_dta))

    ##### Plotting Results
    plt.show()"""
########################################################################################################################
##### CALLING MAIN
if __name__ == "__main__":
    main()