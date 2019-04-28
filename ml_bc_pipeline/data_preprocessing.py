import sys
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from datetime import datetime



class Processor:
    """ Performs data preprocessing

        The objective of this class is to preprocess the data based on training subset. The
        preprocessing steps focus on constant features removal, missing values treatment and
        outliers removal and imputation.

    """
    def __init__(self, training, unseen, outlier_removal):
        """ Constructor

            It is worth to notice that both training and unseen are nothing more nothing less
            than pointers, i.e., pr.training is DF_train and pr.unseen is DF_unseen yields True.
            If you want them to be copies of respective objects, use .copy() on each parameter.

        """
        # Getting the training and unseen data
        self.training = training #.copy() to mantain a copy of the object
        self.unseen = unseen #.copy() to mantain a copy of the object

        # Categorical list of features
        self.feat_c = ["Education", "Marital_Status", "Kidhome", "Teenhome", "AcceptedCmp1", "AcceptedCmp2",
                       "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain"]


        # Both of these seem useless, but we left here for now.
        self._drop_constant_features()
        self._drop_categorical_missing_values()


        # Input missing values of Income and Year_Birth (weird values) with Linear Regression Model
        self._impute_missings_income_regression()
        self._impute_wrong_age_regression()

        # Performs either uni or multivariate outlier detection and inputation/removal
        if (outlier_removal == "uni"):
            # Filtering by 3 std all numerical features and inputting the missings with Linear Regression Model
            self._filter_df_by_std_with_lr_input()
        if(outlier_removal == "multi"):
            # Filters out multivariate outliers through Mahalanobis Distance
            self._multivar_outlier_filter()


        # Binning of Age and Income features creating new features
        self._discreet()





    def _drop_constant_features(self):
        """"
            Since we already removed the constant features (drop_metadata_features(self)) in the data_loader, this has
            no use.

            I will do nothing for nothing. We can remove it in the future if we see it really is useless.
        """



    def _drop_categorical_missing_values(self):
        """"
            Drops Missing Values from categorical values, even though I think the only missing values are in Income.
            Just to be sure.
        """
        self.training.dropna(subset=self.feat_c, inplace=True)
        self.unseen.dropna(subset=self.feat_c, inplace=True)



    def _filter_df_by_std_with_lr_input(self):
        """"
            Puts NaN in outliers, that is, values higher or lower than 3 STD's from the mean, then input these NaNs with
            a Linear Regression model with all other features as independent variables.
        """
        print("####################################################################\n")
        print("#        UNIVARIATE OUTLIER DETECTION BY FILTERING 3 STDS\n")
        print("#        INPUTATION WITH LINEAR REGRESSION MODEL\n")
        print("####################################################################\n")

        def _filter_ser_by_std(series_, n_stdev=3.0):
            mean_, stdev_ = series_.mean(), series_.std()
            cutoff = stdev_ * n_stdev
            lower_bound, upper_bound = mean_ - cutoff, mean_ + cutoff
            return [True if i < lower_bound or i > upper_bound else False for i in series_]

        def lr_input(X, feat):
            y = X[feat]
            y = y[-y.isna()]

            X["Marital_Status"] = pd.Categorical(X["Marital_Status"])
            X["Marital_Status"] = X["Marital_Status"].cat.codes

            X["Education"] = pd.Categorical(X["Education"])
            X["Education"] = X["Education"].cat.codes

            x_pred = X[X[feat].isna()]
            x_pred = x_pred.drop(columns=feat)

            X = X[-X[feat].isna()]
            X = X.drop(columns=feat)

            # Linear Regression Model
            reg = LinearRegression().fit(X, y)

            # Predictions
            y_pred = reg.predict(x_pred)

            return y_pred

        num_feat_list = self.training._get_numeric_data().drop(["Response"], axis=1).columns
        for feat in num_feat_list:
            mask = _filter_ser_by_std(self.training[feat], n_stdev=3.0)
            if (len(self.training[feat][mask]) > 0):
                self.training[feat][mask] = np.NaN
                y_pred_ = lr_input(self.training, feat)
                self.training.loc[self.training[feat].isna(), feat] = y_pred_

        return



    def _impute_missings_income_regression(self):
        """"
            Instead of inputing missing values of Income with mean, we use a Linear Regression Model to estimate
            an approximate value to the Income of these observations through the other independent variables.
        """
        # Function to return the predictions of the Linear Regression Model that receives as parameter the dataset
        def lr_input_income(X):
            y = X["Income"]
            y = y[-y.isna()]

            X["Marital_Status"] = pd.Categorical(X["Marital_Status"])
            X["Marital_Status"] = X["Marital_Status"].cat.codes

            X["Education"] = pd.Categorical(X["Education"])
            X["Education"] = X["Education"].cat.codes

            x_pred = X[X.Income.isna()]
            x_pred = x_pred.drop(columns="Income")

            X = X[-X.Income.isna()]
            X = X.drop(columns="Income")

            # Linear Regression Model
            reg = LinearRegression().fit(X, y)

            # Predictions
            y_pred = reg.predict(x_pred)

            return y_pred


        # Checks if there are cases that match the condition and then apply the fucntion and,
        # then, stores the predictions in the missing values
        if (len(self.training[self.training.Income.isna()])>0):
            y_pred_tr = lr_input_income(self.training)
            self.training.loc[self.training.Income.isna(), "Income"] = y_pred_tr
        if (len(self.unseen[self.unseen.Income.isna()]) > 0):
            y_pred_un = lr_input_income(self.unseen)
            self.unseen.loc[self.unseen.Income.isna(), "Income"] = y_pred_un
        else:
            self.unseen["Marital_Status"] = pd.Categorical(self.unseen["Marital_Status"])
            self.unseen["Marital_Status"] = self.unseen["Marital_Status"].cat.codes

            self.unseen["Education"] = pd.Categorical(self.unseen["Education"])
            self.unseen["Education"] = self.unseen["Education"].cat.codes



    def _impute_wrong_age_regression(self):
        """"
            We find outliers in Year_Birth and, if it is too high (>90) we treat it as a missing value and
            replace it with an estimation from a Linear Regression Model.
        """

        # Function to input age higher than 90 with LRM.
        def lr_age_input(X):
            y = X[X["Age"] < 90].Age
            y = y[-y.isna()]

            X["Marital_Status"] = pd.Categorical(X["Marital_Status"])
            X["Marital_Status"] = X["Marital_Status"].cat.codes

            X["Education"] = pd.Categorical(X["Education"])
            X["Education"] = X["Education"].cat.codes

            x_pred = X[X["Age"] >= 90]
            x_pred = x_pred.drop(columns="Age")


            X = X[X["Age"] < 90]
            X = X.drop(columns="Age")

            # Linear Regression Model
            reg = LinearRegression().fit(X, y)

            # Predictions
            y_pred = reg.predict(x_pred)

            return y_pred

        # Checks if there are cases that match the condition and then apply the function and,
        # then, stores the predictions in the missing values
        if (len(self.training[self.training["Age"] >= 90].Age)>0):
            y_pred_tr = lr_age_input(self.training)
            self.training.loc[self.training["Age"] >= 90, "Age"] = y_pred_tr.round()
        if (len(self.unseen[self.unseen["Age"] >= 90].Age) > 0):
            y_pred_un = lr_age_input(self.unseen)
            self.unseen.loc[self.unseen["Age"] >= 90, "Age"] = y_pred_un.round()



    def _discreet(self):
        """"
            Binning of Age and Income features into new categorical features.
        """
        bindisc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy="uniform")
        feature_bin_training = bindisc.fit_transform(self.training['Income'].values[:, np.newaxis])
        feature_bin_unseen = bindisc.fit_transform(self.unseen['Income'].values[:, np.newaxis])
        self.training['Income_d'] = pd.Series(feature_bin_training[:, 0], index=self.training.index)
        self.unseen['Income_d'] = pd.Series(feature_bin_unseen[:, 0], index=self.unseen.index)

        bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
        feature_bin_training = bindisc.fit_transform(self.training['Age'].values[:, np.newaxis])
        feature_bin_unseen = bindisc.fit_transform(self.unseen['Age'].values[:, np.newaxis])
        self.training['Age_d'] = pd.Series(feature_bin_training[:, 0], index=self.training.index)
        self.unseen['Age_d'] = pd.Series(feature_bin_unseen[:, 0], index=self.unseen.index)



    def _multivar_outlier_filter(self):
        """"
            Detects multivariate outliers through Mahalanobis Distance and removes these rows from the training set.
        """


        print("#####################################################################\n")
        print("#        MULTIVARIATE OUTLIER DETECTION THROUGH MAHALANOBIS DISTANCE\n")
        print("#        DETECTED OUTLIERS REMOVED\n")
        print("#####################################################################\n")

        # Simple function to check if the matrix is positive definite (for example, it will return False if the matrix contains NaN).
        def is_pos_def(A):
            if np.allclose(A, A.T):
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    return False
            else:
                return False

                # The function to calculate the Mahalanobis Distance. Returns a list of distances.

        def MahalanobisDist(data):
            covariance_matrix = np.cov(data, rowvar=False)
            if is_pos_def(covariance_matrix):
                inv_covariance_matrix = np.linalg.inv(covariance_matrix)
                if is_pos_def(inv_covariance_matrix):
                    vars_mean = []
                    for i in range(data.shape[0]):
                        vars_mean.append(list(data.mean(axis=0)))
                    diff = data - vars_mean
                    md = []
                    for i in range(len(diff)):
                        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
                    return md
                else:
                    print("Error: Inverse of Covariance Matrix is not positive definite!")
            else:
                print("Error: Covariance Matrix is not positive definite!")

        # Function to detect multivariate outliers from the Mahalanobis Distances. Returns an array of indexes of the outliers.
        def MD_detectOutliers(data, extreme=False):
            MD = MahalanobisDist(data)

            std = np.std(MD)
            k = 3. * std if extreme else 2. * std
            m = np.mean(MD)
            up_t = m + k
            low_t = m - k
            outliers = []
            for i in range(len(MD)):
                if (MD[i] >= up_t) or (MD[i] <= low_t):
                    outliers.append(i)  # index of the outlier
            return np.array(outliers)

        # Gets the indexes of multivariate outliers
        num_feat_list = self.training._get_numeric_data().drop(["Response", "Education", "Marital_Status"], axis=1).columns
        outliers_i = MD_detectOutliers(np.array(self.training[num_feat_list]))
        # Removes these rows of the training dataset
        self.training = self.training.drop(self.training.index[outliers_i])
        return
