import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_loader2 import Dataset2
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLPC, assess_generalization_auprc, calc_profit, \
    grid_search_NN, grid_search_SVM, grid_search_KNN, grid_search_DTE, grid_search_RF, \
    grid_search_NB, grid_search_LR, grid_search_Bag, grid_search_DT, Voting, Adaboost
from ml_bc_pipeline.utils import BalanceDataset, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import pickle


def predict():
    ########################################################################################################################
    ##### LOAD DATA
    # Choose working directory (path).
    os.chdir("/Users/RodolfoSaldanha/Desktop/ML/project1")  # Change this!
    # Name of the .xlsx file with the data.
    file_path = os.getcwd() + "/ml_project1_data.xlsx"
    unseen = os.getcwd() + "/unseen_students.xlsx"
    # Load the data through the class Dataset.
    ds = Dataset(file_path)  # from data_leader.py
    dsu = Dataset2(unseen)

########################################################################################################################
################ SETUP #################################################################################################
########################################################################################################################
    # Select "uni" or "multi" variate outlier detection.
    # If anything else, will not perform any outlier detection.
    outliers = "uni"

    # Choose Models to run from the list:
    # ["mlp", "nn", "svm", "knn", "dt", "rf", "nb", "lr", "vote", "bag", "ada"]
    #
    # NOTE: "ada" and "vote" need "nb" and "lr" to also be running.
    model_list = ["mlpc", "svm", "knn", "dt", "rf", "nb", "lr", "bag", "vote"]
    #model_list = ["nb", "vote"]

    # Set the seeds.
    #
    #seeds = [12345, 9876]
    seeds = np.random.randint(10000, size=1) # If random seeds are needed.


    # Set the treshold for class assignment (for soft-decision).
    treshold = 0.5

    # Creating lists for storing results:
    results_model = []
    results_auprc = []
    results_precision = []
    results_recal = []
    results_param = []
    results_seeds = []
    results_profit = []

    #####################
    ## !! IMPORTANT !! ##
    # Set a name for the excel files have in this run:
    name = "test_01"

    results_path = os.getcwd() + "/results"
    ensure_dir(results_path)
    os.chdir(results_path)

    results_v_path = os.getcwd() + "/"+name
    ensure_dir(results_v_path)
    os.chdir(results_v_path)

    for seed in seeds:
        ########################################################################################################################
        ##### SPLIT IN TRAIN AND UNSEEN
        # Set seed.
        # DF_train, DF_unseen = train_test_split(ds.rm_df.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
        #                                       random_state=seed)
        DF_train = ds.rm_df.copy()
        DF_unseen = dsu.rm_df.copy()
        ######################################

        ########################################################################################################################
        ##### PREPROCESS
        # The preprocess and feature engineering implemented here is based on information gathered on Feature Exploration
        # Jupyter Notebook.
        pr = Processor(DF_train, DF_unseen, outliers)  # from data_preprocessing.py
        ########################################################################################################################
        ##### FEATURE ENGINEERING
        fe = FeatureEngineer(pr.training, pr.unseen)  # from feature_engineering.py

        # Apply Box-Cox Transformations and gather the best ones for each feature.
        cat = ['Education', 'Marital_Status', 'NmbAccCmps', 'Response', 'Age_d', 'Income_d']
        num_features = fe.training._get_numeric_data().drop(cat, axis=1).columns
        fe.box_cox_transformations(num_features, target="Response")

        # After creating all new features through feature enginnering and Box-Cox transformations,
        # execute the inputation of Missing Values from data_preprocessing.py in case there is some
        # missing from dividing by zero or some weird interaction. Executed both on training and
        # unseen data, just as the Box-Cox Transformations.
        fe._input_missing_values()

        # Rank input features according to Chi-Squared and Decision Tree Algorithm
        continuous_flist = fe.box_cox_features
        categorical_flist = ["Kidhome", "Teenhome", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                             "AcceptedCmp4", "AcceptedCmp5", "Complain", "HasOffspring", 'DT_Acc_1',
                             'DT_MS_Single', 'DT_MS_Widow', 'DT_MS_Divorced', 'DT_E_Phd', 'DT_E_Master',
                             "DT_Age_4", "DT_Age_3", "DT_Age_2", "DT_Age_1", "DT_Income_3", "DT_Income_2",
                             "DT_Income_1"]

        # Remove BxCxT_Recency because it's -0.9 correlated to BxCxT_RFM, which was originated from BxCxT_Recency
        continuous_flist.remove('BxCxT_Recency')

        # Chi-Square Rank
        fe.rank_features_chi_square(continuous_flist, categorical_flist)
        # DTA Rank
        fe.calc_dta_feat_worth(continuous_flist + categorical_flist, 5, 100, 10, seed)
        print("Ranked input features:\n", fe._rank)
        # print(len(continuous_flist + categorical_flist))

        # Feature Selection
        # Selection criteria and number of top features:
        # criteria = ["chisq" or "dta]
        # n_top = [from 1 to inf+]

        DF_train_top_chi, DF_unseen_chi = fe.get_top(criteria="chisq", n_top=47)
        DF_train_top_dta, DF_unseen_dta = fe.get_top(criteria="dta", n_top=10)

        # corr = pd.DataFrame(DF_train_top_chi.corr())
        # corr.to_csv('bla.csv')
        # corr.to_csv('bla.xls')
    ########################################################################################################################
    ################ MODELS ################################################################################################
    ########################################################################################################################

        ####################################################################################################################
        ##### MODEL - MLPC
        if ("mlpc" in model_list):

            mlpc_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                              'mlpc__learning_rate_init': [0.001, 0.01]}


            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)


            mlpc_gscv_chi = grid_search_MLPC(DF_train_top, mlpc_param_grid, seed)
            auprc_mlpc_chi = assess_generalization_auprc(mlpc_gscv_chi.best_estimator_, DF_unseen_top)
            profit_mlpc_chi = calc_profit(mlpc_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(mlpc_gscv_chi.cv_results_).to_csv("mlpc_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            mlpc_gscv_dta = grid_search_MLPC(DF_train_top, mlpc_param_grid, seed)
            auprc_mlpc_dta = assess_generalization_auprc(mlpc_gscv_dta.best_estimator_, DF_unseen_top)
            profit_mlpc_dta = calc_profit(mlpc_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(mlpc_gscv_dta.cv_results_).to_csv("mlpc_gscv_dta_" + str(seed) + ".csv")


            ##### Printing Results

            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", mlpc_gscv_dta.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_mlpc_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", mlpc_gscv_chi.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_mlpc_dta))

            results_model.append("MPL_Chi")
            results_model.append("MPL_DTA")
            results_auprc.append(auprc_mlpc_chi)
            results_auprc.append(auprc_mlpc_dta)
            results_param.append(mlpc_gscv_chi.best_params_)
            results_param.append(mlpc_gscv_dta.best_params_)
            results_profit.append(profit_mlpc_chi)
            results_profit.append(profit_mlpc_dta)



        ########################################################################################################################
        ##### MODEL - NN
        if ("nn" in model_list):

            nn_param_grid = {'nn__batch_size': [25, 32, 40],
                             'nn__epochs': [100]}

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            nn_gscv_chi = grid_search_NN(DF_train_top, nn_param_grid, "chi")
            auprc_nn_chi = assess_generalization_auprc(nn_gscv_chi.best_estimator_, DF_unseen_top)
            profit_nn_chi = calc_profit(nn_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(nn_gscv_chi.cv_results_).to_csv("nn_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection
            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)
            """
            nn_gscv_dta = grid_search_NN(DF_train_top, nn_param_grid, "dta")
            auprc_nn_dta = assess_generalization_auprc(nn_gscv_dta.best_estimator_, DF_unseen_top)
            profit_nn_dta = calc_profit(nn_gscv_dta.best_estimator_, DF_unseen_top, treshold)"""
            ###########
            nn_gscv_dta = nn_gscv_chi
            auprc_nn_dta = auprc_nn_chi
            profit_nn_dta = profit_nn_chi
            ###########
            pd.DataFrame.from_dict(nn_gscv_dta.cv_results_).to_csv("nn_gscv_dta_" + str(seed) + ".csv")

            ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", nn_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_nn_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", nn_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_nn_dta))

            results_model.append("NN_Chi")
            results_model.append("NN_DTA")
            results_auprc.append(auprc_nn_chi)
            results_auprc.append(auprc_nn_dta)
            results_param.append(nn_gscv_chi.best_params_)
            results_param.append(nn_gscv_dta.best_params_)
            results_profit.append(profit_nn_chi)
            results_profit.append(profit_nn_dta)


        ########################################################################################################################
        ##### MODEL - SVM
        if ("svm" in model_list):

            svm_param_grid = {'svm__C': [5,10,15],
                                'svm__kernel': ['linear','rbf']}

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            svm_gscv_chi = grid_search_SVM(DF_train_top, svm_param_grid, seed)
            auprc_svm_chi = assess_generalization_auprc(svm_gscv_chi.best_estimator_, DF_unseen_top)
            profit_svm_chi = calc_profit(svm_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(svm_gscv_chi.cv_results_).to_csv("svm_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            svm_gscv_dta = grid_search_SVM(DF_train_top, svm_param_grid, seed)
            auprc_svm_dta = assess_generalization_auprc(svm_gscv_dta.best_estimator_, DF_unseen_top)
            profit_svm_dta = calc_profit(svm_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(svm_gscv_dta.cv_results_).to_csv("svm_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", svm_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_svm_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", svm_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_svm_dta))

            results_model.append("SVM_Chi")
            results_model.append("SVM_DTA")
            results_auprc.append(auprc_svm_chi)
            results_auprc.append(auprc_svm_dta)
            results_param.append(svm_gscv_chi.best_params_)
            results_param.append(svm_gscv_dta.best_params_)
            results_profit.append(profit_svm_chi)
            results_profit.append(profit_svm_dta)

        ########################################################################################################################
        ##### MODEL - KNN
        if ("knn" in model_list):

            knn_param_grid = {'knn__n_neighbors':[5,6,7,8,9,10],
                            'knn__leaf_size':[1,2,3,5],
                            'knn__weights':['uniform', 'distance'],
                            'knn__algorithm':['auto', 'ball_tree','kd_tree','brute'],
                            'knn__n_jobs':[-1]}

            ##### Chisq Feature Selection


            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            knn_gscv_chi = grid_search_KNN(DF_train_top, knn_param_grid, seed)
            auprc_knn_chi = assess_generalization_auprc(knn_gscv_chi.best_estimator_, DF_unseen_top)
            profit_knn_chi = calc_profit(knn_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(knn_gscv_chi.cv_results_).to_csv("knn_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            knn_gscv_dta = grid_search_KNN(DF_train_top, knn_param_grid, seed)
            auprc_knn_dta = assess_generalization_auprc(knn_gscv_dta.best_estimator_, DF_unseen_top)
            profit_knn_dta = calc_profit(knn_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(knn_gscv_dta.cv_results_).to_csv("knn_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", knn_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_knn_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", knn_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_knn_dta))

            results_model.append("KNN_Chi")
            results_model.append("KNN_DTA")
            results_auprc.append(auprc_knn_chi)
            results_auprc.append(auprc_knn_dta)
            results_param.append(knn_gscv_chi.best_params_)
            results_param.append(knn_gscv_dta.best_params_)
            results_profit.append(profit_knn_chi)
            results_profit.append(profit_knn_dta)


        ########################################################################################################################
        ##### MODEL - DT
        if ("dt" in model_list):

            dt_param_grid = {'dt__max_features': ['auto', 'sqrt', 'log2'],
                              'dt__min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                              'dt__min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}

            ##### Chisq Feature Selection


            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            dt_gscv_chi = grid_search_DT(DF_train_top, dt_param_grid, seed)
            auprc_dt_chi = assess_generalization_auprc(dt_gscv_chi.best_estimator_, DF_unseen_top)
            profit_dt_chi = calc_profit(dt_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(dt_gscv_chi.cv_results_).to_csv("dt_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            dt_gscv_dta = grid_search_DT(DF_train_top, dt_param_grid, seed)
            auprc_dt_dta = assess_generalization_auprc(dt_gscv_dta.best_estimator_, DF_unseen_top)
            profit_dt_dta = calc_profit(dt_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(dt_gscv_dta.cv_results_).to_csv("dt_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", dt_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_dt_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", dt_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_dt_dta))

            results_model.append("DT_Chi")
            results_model.append("DT_DTA")
            results_auprc.append(auprc_dt_chi)
            results_auprc.append(auprc_dt_dta)
            results_param.append(dt_gscv_chi.best_params_)
            results_param.append(dt_gscv_dta.best_params_)
            results_profit.append(profit_dt_chi)
            results_profit.append(profit_dt_dta)

        ########################################################################################################################
        ##### MODEL - DTE

        if ("dte" in model_list):

            dte_param_grid = {'dte__max_features': ['auto', 'sqrt', 'log2'],
                              'dte__min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                              'dte__min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}

            ##### Chisq Feature Selection


            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            dte_gscv_chi = grid_search_DTE(DF_train_top, dte_param_grid, seed)
            auprc_dte_chi = assess_generalization_auprc(dte_gscv_chi.best_estimator_, DF_unseen_top)
            profit_dte_chi = calc_profit(dte_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(dte_gscv_chi.cv_results_).to_csv("dte_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            dte_gscv_dta = grid_search_DTE(DF_train_top, dte_param_grid, seed)
            auprc_dte_dta = assess_generalization_auprc(dte_gscv_dta.best_estimator_, DF_unseen_top)
            profit_dte_dta = calc_profit(dte_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(dte_gscv_dta.cv_results_).to_csv("dte_gscv_dta_" + str(seed) + ".csv")

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ",
                  dte_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_dte_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ",
                  dte_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_dte_dta))

        ########################################################################################################################

            results_model.append("DTE_Chi")
            results_model.append("DTE_DTA")
            results_auprc.append(auprc_dte_chi)
            results_auprc.append(auprc_dte_dta)
            results_param.append(dte_gscv_chi.best_params_)
            results_param.append(dte_gscv_dta.best_params_)
            results_profit.append(profit_dte_chi)
            results_profit.append(profit_dte_dta)

        ########################################################################################################################
        ##### MODEL - RF
        if ("rf" in model_list):

            rf_param_grid = {'rf__criterion':['gini','entropy'],
                            'rf__n_estimators':[10,15,20,25,30],
                            'rf__min_samples_leaf':[1,2,3],
                            'rf__min_samples_split':[3,4,5,6,7]}

            ##### Chisq Feature Selection


            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            rf_gscv_chi = grid_search_RF(DF_train_top, rf_param_grid, seed)
            auprc_rf_chi = assess_generalization_auprc(rf_gscv_chi.best_estimator_, DF_unseen_top)
            profit_rf_chi = calc_profit(rf_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(rf_gscv_chi.cv_results_).to_csv("rf_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            rf_gscv_dta = grid_search_RF(DF_train_top, rf_param_grid, seed)
            auprc_rf_dta = assess_generalization_auprc(rf_gscv_dta.best_estimator_, DF_unseen_top)
            profit_rf_dta = calc_profit(rf_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(rf_gscv_dta.cv_results_).to_csv("rf_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", rf_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_rf_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", rf_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_rf_dta))

            results_model.append("RF_Chi")
            results_model.append("RF_DTA")
            results_auprc.append(auprc_rf_chi)
            results_auprc.append(auprc_rf_dta)
            results_param.append(rf_gscv_chi.best_params_)
            results_param.append(rf_gscv_dta.best_params_)
            results_profit.append(profit_rf_chi)
            results_profit.append(profit_rf_dta)


        ########################################################################################################################
        ##### MODEL - NB
        if ("nb" in model_list):

            nb_param_grid = {}

            ##### Chisq Feature Selection

            print('------------')
            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=DF_train_top_chi["Response"],
                                                                                          random_state=seed)

            nb_gscv_chi = grid_search_NB(DF_train_top, nb_param_grid, seed)
            auprc_nb_chi = assess_generalization_auprc(nb_gscv_chi.best_estimator_, DF_unseen_top)
            profit_nb_chi = calc_profit(nb_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(nb_gscv_chi.cv_results_).to_csv("nb_gscv_chi_" + str(seed) + ".csv")
            print('------------')
            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=DF_train_top_dta["Response"],
                                                                                          random_state=seed)
            print('------------')
            nb_gscv_dta = grid_search_NB(DF_train_top, nb_param_grid, seed)
            auprc_nb_dta = assess_generalization_auprc(nb_gscv_dta.best_estimator_, DF_unseen_top)
            profit_nb_dta = calc_profit(nb_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(nb_gscv_dta.cv_results_).to_csv("nb_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", nb_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_nb_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", nb_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_nb_dta))

            results_model.append("NB_Chi")
            results_model.append("NB_DTA")
            results_auprc.append(auprc_nb_chi)
            results_auprc.append(auprc_nb_dta)
            results_param.append(nb_gscv_chi.best_params_)
            results_param.append(nb_gscv_dta.best_params_)
            results_profit.append(profit_nb_chi)
            results_profit.append(profit_nb_dta)

        ########################################################################################################################
        ##### MODEL - LR
        if ("lr" in model_list):

            lr_param_grid = {}

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            lr_gscv_chi = grid_search_LR(DF_train_top, lr_param_grid, seed)
            auprc_lr_chi = assess_generalization_auprc(lr_gscv_chi.best_estimator_, DF_unseen_top)
            profit_lr_chi = calc_profit(lr_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(lr_gscv_chi.cv_results_).to_csv("lr_gscv_chi_"+str(seed)+".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            lr_gscv_dta = grid_search_LR(DF_train_top, lr_param_grid, seed)
            auprc_lr_dta = assess_generalization_auprc(lr_gscv_dta.best_estimator_, DF_unseen_top)
            profit_lr_dta = calc_profit(lr_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            # Exports raw results to excel file.
            pd.DataFrame.from_dict(lr_gscv_dta.cv_results_).to_csv("lr_gscv_dta_"+str(seed)+".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ", lr_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_lr_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ", lr_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_lr_dta))

            results_model.append("LR_Chi")
            results_model.append("LR_DTA")
            results_auprc.append(auprc_lr_chi)
            results_auprc.append(auprc_lr_dta)
            results_param.append(lr_gscv_chi.best_params_)
            results_param.append(lr_gscv_dta.best_params_)
            results_profit.append(profit_lr_chi)
            results_profit.append(profit_lr_dta)

        ########################################################################################################################
        ##### MODEL - Bagging

        if ("bag" in model_list):

            bag_param_grid = {
                'bag__base_estimator__max_depth': [1, 2, 3, 4, 5],
                'bag__max_samples': [0.05, 0.1, 0.2, 0.5]
            }

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_chi.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            bag_gscv_chi = grid_search_Bag(DF_train_top, bag_param_grid, seed)
            auprc_bag_chi = assess_generalization_auprc(bag_gscv_chi.best_estimator_, DF_unseen_top)
            profit_bag_chi = calc_profit(bag_gscv_chi.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(bag_gscv_chi.cv_results_).to_csv("bag_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = train_test_split(DF_train_top_dta.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                                                                          random_state=seed)

            bag_gscv_dta = grid_search_Bag(DF_train_top, bag_param_grid, seed)
            auprc_bag_dta = assess_generalization_auprc(bag_gscv_dta.best_estimator_, DF_unseen_top)
            profit_bag_dta = calc_profit(bag_gscv_dta.best_estimator_, DF_unseen_top, treshold)
            pd.DataFrame.from_dict(bag_gscv_dta.cv_results_).to_csv("bag_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################
            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ",
                  bag_gscv_chi.best_params_)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_bag_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ",
                  bag_gscv_dta.best_params_)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_bag_dta))

            results_model.append("Bag_Chi")
            results_model.append("Bag_DTA")
            results_auprc.append(auprc_bag_chi)
            results_auprc.append(auprc_bag_dta)
            results_param.append(bag_gscv_chi.best_params_)
            results_param.append(bag_gscv_dta.best_params_)
            results_profit.append(profit_bag_chi)
            results_profit.append(profit_bag_dta)

        ########################################################################################################################
        ##### MODEL - Voting
        if ("vote" in model_list):

            if (auprc_nb_dta > auprc_nb_chi):
                nb = nb_gscv_dta
            else:
                nb = nb_gscv_chi
            if (auprc_lr_dta > auprc_lr_chi):
                lr = lr_gscv_dta
            else:
                lr = lr_gscv_chi

            if (auprc_svm_dta > auprc_svm_chi):
                svm = svm_gscv_dta
            else:
                svm = svm_gscv_chi

            if (auprc_mlpc_dta > auprc_mlpc_chi):
                mlpc = mlpc_gscv_dta
            else:
                mlpc = mlpc_gscv_chi
            
            if (auprc_knn_dta > auprc_knn_chi):
                knn = knn_gscv_dta
            else:
                knn = knn_gscv_chi
            if (auprc_dt_dta > auprc_dt_chi):
                dt = dt_gscv_dta
            else:
                dt = dt_gscv_chi
            if (auprc_rf_dta > auprc_rf_chi):
                rf = rf_gscv_dta
            else:
                rf = rf_gscv_chi
            if (auprc_bag_dta > auprc_bag_chi):
                bag = bag_gscv_dta
            else:
                bag = bag_gscv_chi
            
            estimators = [('nb', nb.best_estimator_.named_steps['nb']),
                          ('lr', lr.best_estimator_.named_steps['lr']),
                          ('svm', svm.best_estimator_.named_steps['svm']),
                          ('mlpc', mlpc.best_estimator_.named_steps['mlpc']),
                          ('knn', knn.best_estimator_.named_steps['knn']),
                          ('dt', dt.best_estimator_.named_steps['dt']),
                          ('rf', rf.best_estimator_.named_steps['rf']),
                          ('bag', bag.best_estimator_.named_steps['bag'])]

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            DF_unseen_chi.drop(columns="Response", inplace=True)

            vote_gscv_chi = Voting(DF_train_top, estimators, seed)
            pred = vote_gscv_chi.predict(DF_unseen_top)
            original = dsu.original.copy()
            original.drop(columns="Response", inplace=True)
            pred = pd.DataFrame(pred, columns=["Response"])
            result = pd.concat([original, pred], axis=1)
            result.to_csv("/Users/RodolfoSaldanha/Desktop/final.csv")
            #auprc_vote_chi = assess_generalization_auprc(vote_gscv_chi, DF_unseen_top)
            #profit_vote_chi = calc_profit(vote_gscv_chi, DF_unseen_top, treshold)

            #pd.DataFrame.from_dict(vote_gscv_chi.cv_results_).to_csv("vote_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            vote_gscv_dta = Voting(DF_train_top, estimators, seed)
            auprc_vote_dta = assess_generalization_auprc(vote_gscv_dta, DF_unseen_top)
            profit_vote_dta = calc_profit(vote_gscv_dta, DF_unseen_top, treshold)
            #pd.DataFrame.from_dict(vote_gscv_dta.cv_results_).to_csv("vote_gscv_dta_" + str(seed) + ".csv")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ",
                  vote_gscv_chi)
            #print("Chisq Selection AUPRC: {:.2f}".format(auprc_vote_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ",
                  vote_gscv_dta)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_vote_dta))

            results_model.append("Vote_Chi")
            results_model.append("Vote_DTA")
            #results_auprc.append(auprc_vote_chi)
            results_auprc.append(auprc_vote_dta)
            results_param.append(vote_gscv_chi)
            results_param.append(vote_gscv_dta)
            #results_profit.append(profit_vote_chi)
            results_profit.append(profit_vote_dta)

        ########################################################################################################################
        ##### MODEL - AdaBoost

        if ("ada" in model_list):

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            ada_gscv_chi = Adaboost(DF_train_top, seed)
            auprc_ada_chi = assess_generalization_auprc(ada_gscv_chi, DF_unseen_top)
            profit_ada_chi = calc_profit(ada_gscv_chi, DF_unseen_top, treshold)
            #pd.DataFrame.from_dict(ada_gscv_chi.cv_results_).to_csv("ada_gscv_chi_" + str(seed) + ".csv")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            ada_gscv_dta = Adaboost(DF_train_top, seed)
            auprc_ada_dta = assess_generalization_auprc(ada_gscv_dta, DF_unseen_top)
            profit_ada_dta = calc_profit(ada_gscv_dta, DF_unseen_top, treshold)
            #pd.DataFrame.from_dict(ada_gscv_dta.cv_results_).to_csv("ada_gscv_dta_" + str(seed) + ".csv")

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ",
                  ada_gscv_chi)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_ada_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ",
                  ada_gscv_dta)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_ada_dta))

        ########################################################################################################################
            results_model.append("Ada_Chi")
            results_model.append("Ada_DTA")
            results_auprc.append(auprc_ada_chi)
            results_auprc.append(auprc_ada_dta)
            results_param.append(ada_gscv_chi)
            results_param.append(ada_gscv_dta)
            results_profit.append(profit_ada_chi)
            results_profit.append(profit_ada_dta)
        ########################################################################################################################


        # Storing seeds to show with results
        for ind in range(len(model_list)*2):
            results_seeds.append(seed)

    ########################
    ## SUMMARY OF RESULTS ##
    print(results_seeds)
    print(results_model)
    print(results_param)
    print(results_auprc)
    print(results_profit)
    print(results_recal)
    print(results_precision)


    #####################################################
    # Creates a folder to hold the summary excel files; #
    # Export summary results to excel.                  #

    summary_path = os.getcwd() + "summary"
    ensure_dir(summary_path)
    os.chdir(summary_path)




    ##### Plotting Results
    #plt.show()
########################################################################################################################
##### CALLING MAIN
