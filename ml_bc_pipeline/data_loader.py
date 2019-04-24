import sys
from typing import List

import pandas as pd
from datetime import datetime, date


class Dataset:
    """ Loads and prepares the data

        The objective of this class is load the dataset and execute basic data
        preparation before effectively moving into the cross validation workflow.

    """
    def __init__(self, full_path):
        """"
            The constructor method just executes all of its methods on the dataset.
        """
        self.rm_df = pd.read_excel(full_path, index_col=0)
        self._drop_metadata_features()
        self._drop_doubleback_features()
        self._drop_unusual_classes()
        self._label_encoder()
        self._as_category()
        self._days_since_customer()

    def _drop_metadata_features(self):
        """"
            Here we remove the constant features used only to, later, calculate cost and revenue of an accepted
            campaign.
        """

        metadata_features = ['Z_CostContact', 'Z_Revenue']
        self.rm_df.reset_index(drop=True, inplace=True)
        self.rm_df.drop(labels=metadata_features, axis=1, inplace=True)


    def _drop_doubleback_features(self):
        """ Drops perfectly correlated feature

            Since the only high correlation we found was between MntMeatProducts and NumCatalogPurchases using the first
            seed, and between both of them and Income too using the second seed, we will NOT be removing any of them because
            conceptually there is not a intrinsic relationship between them that other features do not also have.

            So this method will do nothing until we find a problematic correlation.
        """

        #self.rm_df.drop(["NetPurchase"], axis=1, inplace=True)

    def _drop_unusual_classes(self):
        """"
            This drops all observations of the dataset that contain these weird classes of Marital_Status.
        """
        errors_dict = ["YOLO", "Alone", "Absurd"]
        for value in errors_dict:
            self.rm_df = self.rm_df[self.rm_df["Marital_Status"] != value]


    def _label_encoder(self):
        """ Manually encodes categories (labels) in the categorical features

            You could use automatic label encoder from sklearn (sklearn.preprocessing.LabelEncoder), however,
            when it is possible, I prefer to use a manual encoder such that I have a control on the code of
            each label. This makes things easier to interpret when analyzing the outcomes of our ML algorithms.

        """

        cleanup_nums = {
                        "Education": {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4},
                        "Marital_Status": {'Single': 0, 'Widow': 1, 'Divorced': 2, 'Married': 3, 'Together': 4}
                        }
        self.rm_df.replace(cleanup_nums, inplace=True)


    def _as_category(self):
        """
            Explicitly encodes all categorical features as categories
        """

        feat_c = ["Education", "Marital_Status", "Kidhome", "Teenhome", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                  "AcceptedCmp4", "AcceptedCmp5", "Complain"]

        for feat in feat_c:
            self.rm_df[feat] = self.rm_df[feat].astype('category')


    def _days_since_customer(self):
        """ Encodes Dt_Customer (nº days since customer)

            Similarly to the label encoder, we have to transform the Dt_Customer in order to feed numerical
            quantities into our ML algorithms. Here we encode Dt_Customer into number the of days since, for
            example, first of April of 2019 (01/04/2019).

        """

        # Gets a series of dates and its format as parameters and returns a series of days since that date until today.
        date_format = "%Y-%m-%d"

        self.rm_df["Dt_Customer"] = self.rm_df["Dt_Customer"].apply(lambda x: (datetime.today()-datetime.strptime(x, date_format)).days)

