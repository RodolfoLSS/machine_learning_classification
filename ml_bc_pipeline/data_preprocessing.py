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
    def __init__(self, training, unseen):
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

        # For now we will not be filtering out the outliers by std.
        #num_features = self._filter_df_by_std()
        #self._impute_num_missings_mean(num_features)
        self._age_transformation()

        # Input missing values of Income and Year_Birth (weird values) with Linear Regression Model
        self._impute_missings_income_regression()
        self._impute_wrong_age_regression()
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

    def _filter_df_by_std(self):
        """"
            Inputs NaN in outliers, that is, values higher or lower than 3 STD's from the mean.
        """
        def _filter_ser_by_std(series_, n_stdev=3.0):
            mean_, stdev_ = series_.mean(), series_.std()
            cutoff = stdev_ * n_stdev
            lower_bound, upper_bound = mean_ - cutoff, mean_ + cutoff
            return [True if i < lower_bound or i > upper_bound else False for i in series_]

        training_num = self.training._get_numeric_data().drop(["Response"], axis=1)
        mask = training_num.apply(axis=0, func=_filter_ser_by_std, n_stdev=3.0)
        training_num[mask] = np.NaN
        self.training[training_num.columns] = training_num

        return list(training_num.columns)

    def _impute_num_missings_mean(self, num_features):
        """"
            Use the mean to input missing values into numeric variables.
        """
        self._imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train_imputed = self._imputer.fit_transform(self.training[num_features].values)
        X_unseen_imputed = self._imputer.transform(self.unseen[num_features].values)

        self.training[num_features] = X_train_imputed
        self.unseen[num_features] = X_unseen_imputed

    def _age_transformation(self):
        """"
            Use the mean to input missing values into numeric variables.
        """
        self.training['Age'] = 2019 - self.training['Year_Birth']
        self.unseen['Age'] = 2019 - self.unseen['Year_Birth']

        self.training.drop(columns="Year_Birth", inplace=True)
        self.unseen.drop(columns="Year_Birth", inplace=True)

    def _impute_missings_income_regression(self):
        """"
            Instead of inputing missing values of Income with mean, we use a Linear Regression Model to estimate
            an approximate value to the Income of these observations through the other independent variables.
        """
        # Function to return the predictions of the Linear Regression Model that receives as parameter the dataset
        def lr_input(X):
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
            y_pred_tr = lr_input(self.training)
            self.training.loc[self.training.Income.isna(), "Income"] = y_pred_tr
        if (len(self.unseen[self.unseen.Income.isna()]) > 0):
            y_pred_un = lr_input(self.unseen)
            self.unseen.loc[self.unseen.Income.isna(), "Income"] = y_pred_un

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

        # Checks if there are cases that match the condition and then apply the fucntion and,
        # then, stores the predictions in the missing values
        if (len(self.training[self.training["Age"] >= 90].Age)>0):
            y_pred_tr = lr_age_input(self.training)
            self.training.loc[self.training["Age"] >= 90, "Age"] = y_pred_tr.round()
        if (len(self.unseen[self.unseen["Age"] >= 90].Age) > 0):
            y_pred_un = lr_age_input(self.unseen)
            self.unseen.loc[self.unseen["Age"] >= 90, "Age"] = y_pred_un.round()


    def _discreet(self):
        """"
            Method to rank all features according to chi-square test for independence in relation to Response.
            All based solely on the training set.
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