import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auprc, \
    grid_search_NN, grid_search_SVM, grid_search_KNN, grid_search_DTE, grid_search_RF, \
    grid_search_NB, grid_search_LR, grid_search_Bag, grid_search_DT, Voting, Adaboost
from ml_bc_pipeline.utils import BalanceDataset
import numpy as np
import matplotlib.pyplot as plt


def main():
    ########################################################################################################################
    ##### LOAD DATA
    # Choose working directory (path).
    os.chdir("/Users/RodolfoSaldanha/Desktop/ML/project1")  # Change this!
    # Name of the .xlsx file with the data.
    file_path = os.getcwd() + "/ml_project1_data.xlsx"
    # Load the data through the class Dataset.
    ds = Dataset(file_path)  # from data_leader.py

########################################################################################################################
################ SETUP #################################################################################################
########################################################################################################################
    # Select "uni" or "multi" variate outlier detection.
    # If anything else, will not perform any outlier detection.
    outliers = "multi"

    # Choose Models to run from the list:
    # ["mlp", "nn", "svm", "knn", "dt", "rf", "nb", "lr"]
    model_list = ["nb", "lr", "mlp", "nn", "vote"]

    # Set the seeds.
    #
    #seeds = [12345, 9876]
    seeds = np.random.randint(10000, size=2) # If random seeds are needed.

    # Creating lists for storing results:
    results_model = []
    results_auprc = []
    results_param = []
    results_seeds = []




    for seed in seeds:
        ########################################################################################################################
        ##### SPLIT IN TRAIN AND UNSEEN
        # Set seed.
        DF_train, DF_unseen = train_test_split(ds.rm_df.copy(), test_size=0.2, stratify=ds.rm_df["Response"],
                                               random_state=seed)

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
                             "DT_Age_4", "DT_Age_3", "DT_Age_2", "DT_Age_1", "DT_Income_3", "DT_Income_2", "DT_Income_1"]

        # Remove BxCxT_Recency because it's -0.9 correlated to BxCxT_RFM, which was originated from BxCxT_Recency
        continuous_flist.remove('BxCxT_Recency')

        # Chi-Square Rank
        fe.rank_features_chi_square(continuous_flist, categorical_flist)
        # DTA Rank
        fe.calc_dta_feat_worth(continuous_flist + categorical_flist, 5, 100, 10, seed)
        print("Ranked input features:\n", fe._rank)
        # print(len(continuous_flist + categorical_flist))

        # Feature Selection
        # SET selection criteria and number of top features:
        # criteria = ["chisq" or "dta]
        # n_top = [from 1 to inf+]
        # n_top = 44

        DF_train_top_chi, DF_unseen_chi = fe.get_top(criteria="chisq", n_top=47)
        DF_train_top_dta, DF_unseen_dta = fe.get_top(criteria="dta", n_top=15)

        # corr = pd.DataFrame(DF_train_top_chi.corr())
        # corr.to_csv('bla.csv')
        # corr.to_excel('bla.xls')

    ########################################################################################################################
    ################ MODELS ################################################################################################
    ########################################################################################################################

        ####################################################################################################################
        ##### MODEL - MLP
        if ("mlp" in model_list):

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

            results_model.append("MPL_Chi")
            results_model.append("MPL_DTA")
            results_auprc.append(auprc_mlp_chi)
            results_auprc.append(auprc_mlp_dta)
            results_param.append(mlp_gscv_chi.best_params_)
            results_param.append(mlp_gscv_dta.best_params_)



        ########################################################################################################################
        ##### MODEL - NN
        if ("nn" in model_list):

            nn_param_grid = {'nn__batch_size': [25, 32, 40],
                             'nn__epochs': [100]}

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            nn_gscv_chi = grid_search_NN(DF_train_top, nn_param_grid, "chi")
            auprc_nn_chi = assess_generalization_auprc(nn_gscv_chi.best_estimator_, DF_unseen_top)
            ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

            ##### DTA Feature Selection
            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            nn_gscv_dta = grid_search_NN(DF_train_top, nn_param_grid, "dta")
            auprc_nn_dta = assess_generalization_auprc(nn_gscv_dta.best_estimator_, DF_unseen_top)
            # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

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


        ########################################################################################################################
        ##### MODEL - SVM
        if ("svm" in model_list):

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

            results_model.append("SVM_Chi")
            results_model.append("SVM_DTA")
            results_auprc.append(auprc_svm_chi)
            results_auprc.append(auprc_svm_dta)
            results_param.append(svm_gscv_chi.best_params_)
            results_param.append(svm_gscv_dta.best_params_)

        ########################################################################################################################
        ##### MODEL - KNN
        if ("knn" in model_list):

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

            results_model.append("KNN_Chi")
            results_model.append("KNN_DTA")
            results_auprc.append(auprc_knn_chi)
            results_auprc.append(auprc_knn_dta)
            results_param.append(knn_gscv_chi.best_params_)
            results_param.append(knn_gscv_dta.best_params_)


        ########################################################################################################################
        ##### MODEL - DT
        if ("dt" in model_list):

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

            results_model.append("DT_Chi")
            results_model.append("DT_DTA")
            results_auprc.append(auprc_dt_chi)
            results_auprc.append(auprc_dt_dta)
            results_param.append(dt_gscv_chi.best_params_)
            results_param.append(dt_gscv_dta.best_params_)

        ########################################################################################################################
        ##### MODEL - DTE

        if ("dte" in model_list):

            dte_param_grid = {'dte__max_features': ['auto', 'sqrt', 'log2'],
                              'dte__min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                              'dte__min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}

            ##### Chisq Feature Selection


            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            dte_gscv_chi = grid_search_DTE(DF_train_top, dte_param_grid, seed)
            auprc_dte_chi = assess_generalization_auprc(dte_gscv_chi.best_estimator_, DF_unseen_top)
            ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            dte_gscv_dta = grid_search_DTE(DF_train_top, dte_param_grid, seed)
            auprc_dte_dta = assess_generalization_auprc(dte_gscv_dta.best_estimator_, DF_unseen_top)
            # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

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

        ########################################################################################################################
        ##### MODEL - RF
        if ("rf" in model_list):

            rf_param_grid = {'rf__criterion':['gini','entropy'],
                            'rf__n_estimators':[10,15,20,25,30],
                            'rf__min_samples_leaf':[1,2,3],
                            'rf__min_samples_split':[3,4,5,6,7]}

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

            results_model.append("RF_Chi")
            results_model.append("RF_DTA")
            results_auprc.append(auprc_rf_chi)
            results_auprc.append(auprc_rf_dta)
            results_param.append(rf_gscv_chi.best_params_)
            results_param.append(rf_gscv_dta.best_params_)


        ########################################################################################################################
        ##### MODEL - NB
        if ("nb" in model_list):

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

            results_model.append("NB_Chi")
            results_model.append("NB_DTA")
            results_auprc.append(auprc_nb_chi)
            results_auprc.append(auprc_nb_dta)
            results_param.append(nb_gscv_chi.best_params_)
            results_param.append(nb_gscv_dta.best_params_)

        ########################################################################################################################
        ##### MODEL - LR
        if ("lr" in model_list):

            lr_param_grid = {}

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            lr_gscv_chi = grid_search_LR(DF_train_top, lr_param_grid, seed)
            auprc_lr_chi = assess_generalization_auprc(lr_gscv_chi.best_estimator_, DF_unseen_top)
            ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            lr_gscv_dta = grid_search_LR(DF_train_top, lr_param_grid, seed)
            auprc_lr_dta = assess_generalization_auprc(lr_gscv_dta.best_estimator_, DF_unseen_top)
            # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

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

        ########################################################################################################################
        ##### MODEL - Voting
        if ("vote" in model_list):

            estimators = [('nb', nb_gscv_chi.best_estimator_.named_steps['nb']),
                          ('lr', lr_gscv_chi.best_estimator_.named_steps['lr']),
                          ('mlp', mlp_gscv_chi.best_estimator_.named_steps['mlp']),
                          ('nn', nn_gscv_chi.best_estimator_.named_steps['nn'])]

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            vote_gscv_chi = Voting(DF_train_top, estimators, seed)
            auprc_vote_chi = assess_generalization_auprc(vote_gscv_chi, DF_unseen_top)
            ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            vote_gscv_dta = Voting(DF_train_top, estimators, seed)
            auprc_vote_dta = assess_generalization_auprc(vote_gscv_dta, DF_unseen_top)
            # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

        ########################################################################################################################

            ##### Printing Results
            print("\n--------> CHISQ FEATURE SELECTION:\nBest parameter set with Chisq Selection: ",
                  vote_gscv_chi)
            print("Chisq Selection AUPRC: {:.2f}".format(auprc_vote_chi))

            print("\n--------> DTA FEATURE SELECTION:\nBest parameter set with DTA Selection: ",
                  vote_gscv_dta)
            print("DTA Selection AUPRC: {:.2f}".format(auprc_vote_dta))

            results_model.append("Vote_Chi")
            results_model.append("Vote_DTA")
            results_auprc.append(auprc_vote_chi)
            results_auprc.append(auprc_vote_dta)
            results_param.append(vote_gscv_chi)
            results_param.append(vote_gscv_dta)

        ########################################################################################################################
        ##### MODEL - Bagging
        if ("bag" in model_list):

            bag_param_grid = {
                'bag__base_estimator__max_depth': [1, 2, 3, 4, 5],
                'bag__max_samples': [0.05, 0.1, 0.2, 0.5]
            }

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            bag_gscv_chi = grid_search_Bag(DF_train_top, bag_param_grid, seed)
            auprc_bag_chi = assess_generalization_auprc(bag_gscv_chi.best_estimator_, DF_unseen_top)
            ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            bag_gscv_dta = grid_search_Bag(DF_train_top, bag_param_grid, seed)
            auprc_bag_dta = assess_generalization_auprc(bag_gscv_dta.best_estimator_, DF_unseen_top)
            # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

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

        ########################################################################################################################
        ##### MODEL - AdaBoost

        if ("ada" in model_list):

            estimators = [('nb', nb_gscv_chi.best_estimator_.named_steps['nb']),
                          ('lr', lr_gscv_chi.best_estimator_.named_steps['lr'])]

            ##### Chisq Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_chi, DF_unseen_chi

            ada_gscv_chi = Adaboost(DF_train_top, estimators, seed)
            auprc_ada_chi = assess_generalization_auprc(ada_gscv_chi, DF_unseen_top)
            ###pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_chisq.xlsx")

            ##### DTA Feature Selection

            DF_train_top, DF_unseen_top = DF_train_top_dta, DF_unseen_dta

            ada_gscv_dta = Adaboost(DF_train_top, estimators, seed)
            auprc_ada_dta = assess_generalization_auprc(ada_gscv_dta, DF_unseen_top)
            # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("mlp_gscv_dta.xlsx")

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


        for ind in range(len(model_list)*2):
            results_seeds.append(seed)

    ########################
    ## SUMMARY OF RESULTS ##
    best_param = pd.DataFrame({"Seed" : results_seeds, "Model" : results_model, "Best Parameters" : results_param})

    results = pd.DataFrame({"Seed" : results_seeds, "Model" : results_model, "AUPRC" : results_auprc})

    avgs = results.groupby(['Model']).mean()["AUPRC"]
    result_avg = pd.DataFrame({"Avg. AUPRC" : avgs})

    # Printing it all
    print("\n\n----------------------------------------------------------------")
    print("Best Parameters for each seed:\n")
    print(best_param)
    print("\n----------------------------------------------------------------")
    print("Best Results for each seed:\n")
    print(results)
    print("\n----------------------------------------------------------------")
    print("Average Results through all seeds:\n")
    print(result_avg)
    print("\n----------------------------------------------------------------")


    ##### Plotting Results
    #plt.show()
########################################################################################################################
##### CALLING MAIN
if __name__ == "__main__":
    main()