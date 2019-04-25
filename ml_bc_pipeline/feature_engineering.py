import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA


class FeatureEngineer:
    def __init__(self, training, unseen):
        self._rank = {}
        self.training = training
        self.unseen = unseen

        self._extract_business_features()
        self._merge_categories()
        self._generate_dummies()



    def _extract_business_features(self):
        """"
            Here we extract the new business oriented featured from the original features
        """

        # Put all of the creation of new variables into a function so we can call it both for self.training and
        # for self.unseen instead of writing all this twice.
        def create_bus_feat(df):
            n = df.shape[0]

            # Percentage of Monetary Units spent on gold products out of the total spent
            aux = [0] * n

            for i in range(n):
                aux[i] = df["MntGoldProds"].iloc[i] / sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :]) * 100

            df["PrpGoldProds"] = aux

            # Number of Accepted Campaigns out of the last 5 Campaigns
            aux = [0] * n

            for i in range(n):
                aux[i] = sum(
                    df[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i, :])

            df["NmbAccCmps"] = aux

            # Proportion of Accepted Campaigns out of the last 5 Campaigns
            aux = [0] * n

            for i in range(n):
                aux[i] = (sum(
                    df[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,
                    :]) / 5) * 100

            df["PrpAccCmps"] = aux

            # Proportion of Monetary Units spent on Wine out of the total spent
            aux = [0] * n

            for i in range(n):
                aux[i] = float(df[["MntWines"]].iloc[i, :] / sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :])) * 100

            df["PrpWines"] = aux

            # Proportion of Monetary Units spent on Fruits out of the total spent
            aux = [0] * n

            for i in range(n):
                aux[i] = float(df[["MntFruits"]].iloc[i, :] / sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :])) * 100

            df["PrpFruits"] = aux

            # Proportion of Monetary Units spent on Meat out of the total spent
            aux = [0] * n

            for i in range(n):
                aux[i] = float(df[["MntMeatProducts"]].iloc[i, :] / sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :])) * 100

            df["PrpMeat"] = aux

            # Proportion of Monetary Units spent on Fish out of the total spent
            aux = [0] * n

            for i in range(n):
                aux[i] = float(df[["MntFishProducts"]].iloc[i, :] / sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :])) * 100

            df["PrpFish"] = aux

            # Proportion of Monetary Units spent on Sweets out of the total spent
            aux = [0] * n

            for i in range(n):
                aux[i] = float(df[["MntSweetProducts"]].iloc[i, :] / sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :])) * 100

            df["PrpSweets"] = aux

            # Monetary
            aux = [0] * n

            for i in range(n):
                aux[i] = sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i, :])

            df["Mnt"] = aux

            # Buy Potential
            aux = [0] * n

            for i in range(n):
                aux[i] = float(sum(
                    df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,
                    :]) / ((df[["Income"]].iloc[i, :]) * 2))

            df["BuyPot"] = aux

            # Frequency
            aux = [0] * n

            for i in range(n):
                aux[i] = sum(
                    df[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].iloc[i, :])

            df["Freq"] = aux

            # Creating RFM feature using Recency, Freq and Mnt:
            feature_list, n_bins = ["Recency", "Freq", "Mnt"], 5
            rfb_dict = {}
            for feature in feature_list:
                bindisc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy="quantile")
                feature_bin = bindisc.fit_transform(df[feature].values[:, np.newaxis])
                feature_bin = pd.Series(feature_bin[:, 0], index=df.index)
                feature_bin += 1

                if feature == "Recency":
                    feature_bin = feature_bin.sub(5).abs() + 1
                rfb_dict[feature + "_bin"] = feature_bin.astype(int).astype(str)

            df["RFM"] = (rfb_dict['Recency_bin'] + rfb_dict['Freq_bin'] + rfb_dict['Mnt_bin']).astype(int)

            # Creating new feature using PCA to summarize all features in 2 dimensions (2 new features)
            columns = df.columns
            columns = columns.drop(["Response", "Marital_Status", "Education"])

            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(df[columns])

            df["pc1_"] = principalComponents[:, 0]
            df["pc2_"] = principalComponents[:, 1]

            # Creating new feature using PCA to summarize all features in 5 dimensions (5 new features)
            pca = PCA(n_components=5)
            principalComponents = pca.fit_transform(df[columns])

            df["pc1"] = principalComponents[:, 0]
            df["pc2"] = principalComponents[:, 1]
            df["pc3"] = principalComponents[:, 2]
            df["pc4"] = principalComponents[:, 3]
            df["pc5"] = principalComponents[:, 4]




        create_bus_feat(self.training)
        create_bus_feat(self.unseen)



    def _merge_categories(self):
        """"
            We merge the categories Marital_Status and Education respecting the enconding previously done in data_loader
            It it as follows:
                Marital_Status:  "Single" as 3, "Widow" as 2, "Divorced" as 1 and ["Married", "Together"] as 0
                Education: "Phd" as 2, "Master" as 1 and ['Graduation', 'Basic', '2n Cycle'] as 0

            The feature HasOffspring that works as a kind of merging of KidHome and TeenHome indicating presence
            of offsrping
        """
        self.dict_merge_cat = {"Marital_Status": lambda x: 3 if x == 0 else (2 if x == 1 else (1 if x == 2 else 0)),
                               "Education": lambda x: 2 if x == 4 else (1 if x == 3 else 0),
                               "NmbAccCmps": lambda x: 1 if x > 0 else 0,
                               "Age_d": lambda x: 3 if x == 0 else (2 if x == 1 else (1 if x == 2 else (4 if x == 4 else 0))),
                               "Income_d": lambda x: 3 if x == 5 else (2 if x == 4 else (1 if x == 3 else 0))}

        # Applies the dictionary on both datasets
        for key, value in self.dict_merge_cat.items():
            self.training["MC_"+key] = self.training[key].apply(value).astype('category')
            self.unseen["MC_" + key] = self.unseen[key].apply(value).astype('category')


        # Function to apply both on traininf and unseen that creates the HasOffspring feature that
        # indicates presence of any children or teen
        def create_hasoffsrping(df):
            # HasOffsrping Feature
            aux = [0] * df.shape[0]

            for i in range(df.shape[0]):
                if (int(df[["Kidhome"]].iloc[i, :]) + int(df[["Teenhome"]].iloc[i, :]) > 0):
                    aux[i] = 1
                else:
                    aux[i] = 0

            df["HasOffspring"] = aux
            df["HasOffspring"] = df["HasOffspring"].astype('category')

        # Applies the function create_hasoffspring on both datasets
        create_hasoffsrping(self.training)
        create_hasoffsrping(self.unseen)



    def _generate_dummies(self):
        """"
            Use OneHotEncoding to generate dummies for the merged Marital_Status and Education features
        """
        features_to_enconde = ['MC_Marital_Status', 'MC_Education', 'MC_NmbAccCmps', 'MC_Age_d', 'MC_Income_d']
        columns = ["DT_MS_Single", "DT_MS_Widow", "DT_MS_Divorced", "DT_E_Phd", "DT_E_Master", "DT_Acc_1",
                   "DT_Age_4", "DT_Age_3", "DT_Age_2", "DT_Age_1", "DT_Income_3", "DT_Income_2", "DT_Income_1"]
        idxs = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # 1(Single), 2(Widow), 3(Divorced), 4(MS_Zero), 5(Phd),
                                    # 6(Master), 7(Accepted at least 1 campaign) , 8(Educ_Zero)
        # encode categorical features from training data as a one-hot numeric array.
        enc = OneHotEncoder(handle_unknown='ignore')
        Xtr_enc = enc.fit_transform(self.training[features_to_enconde]).toarray()
        # update training data
        df_temp = pd.DataFrame(Xtr_enc[:, idxs], index=self.training.index, columns=columns)
        self.training = pd.concat([self.training, df_temp], axis=1)
        for c in columns:
            self.training[c] = self.training[c].astype('category')
        # use the same encoder to transform unseen data
        Xun_enc = enc.transform(self.unseen[features_to_enconde]).toarray()
        # update unseen data
        df_temp = pd.DataFrame(Xun_enc[:, idxs], index=self.unseen.index, columns=columns)
        self.unseen = pd.concat([self.unseen, df_temp], axis=1)
        for c in columns:
            self.unseen[c] = self.unseen[c].astype('category')



    def box_cox_transformations(self, num_features, target):
        """"
            Applies the box-cox transformations to the numerical features and checks which transformations are better
            for each feature and appends them to the training and unseen datasets
        """

        # 1) perform feature scaling, using MinMaxScaler from sklearn
        bx_cx_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        X_tr_01 = bx_cx_scaler.fit_transform(self.training[num_features].values)
        X_un_01 = bx_cx_scaler.transform(self.unseen[num_features].values)
        num_features_BxCx = ["BxCxT_" + s for s in num_features]
        self.training = pd.concat([self.training.loc[:, self.training.columns != target],
                                   pd.DataFrame(X_tr_01, index=self.training.index, columns=num_features_BxCx),
                                   self.training[target]], axis=1)
        self.unseen = pd.concat([self.unseen.loc[:, self.unseen.columns != target],
                                   pd.DataFrame(X_un_01, index=self.unseen.index, columns=num_features_BxCx),
                                   self.unseen[target]], axis=1)
        # 2) define a set of transformations
        self._bx_cx_trans_dict = {"x": lambda x: x, "log": np.log, "sqrt": np.sqrt,
                      "exp": np.exp, "**1/4": lambda x: np.power(x, 0.25),
                      "**2": lambda x: np.power(x, 2), "**4": lambda x: np.power(x, 4)}
        # 3) perform power transformations on scaled features and select the best
        self.best_bx_cx_dict = {}
        for feature in num_features_BxCx:
            best_test_value, best_trans_label, best_power_trans = 0, "", None
            for trans_key, trans_value in self._bx_cx_trans_dict.items():
                # 3) 1) 1) apply transformation on training data
                feature_trans = np.round(trans_value(self.training[feature]), 4)
                if trans_key == "log":
                    feature_trans.loc[np.isfinite(feature_trans) == False] = -50
                # 3) 1) 2) bin transformed feature (required to perform Chi-Squared test)
                bindisc = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
                feature_bin = bindisc.fit_transform(feature_trans.values.reshape(-1, 1))
                feature_bin = pd.Series(feature_bin[:, 0], index=self.training.index)
                # 3) 1) 3) obtain contingency table
                cont_tab = pd.crosstab(feature_bin, self.training[target], margins=False)
                # 3) 1) 4) compute Chi-Squared test
                chi_test_value = stats.chi2_contingency(cont_tab)[0]
                # 3) 1) 5) choose the best so far Box-Cox transformation based on Chi-Squared test
                if chi_test_value > best_test_value:
                    best_test_value, best_trans_label, best_power_trans = chi_test_value, trans_key, feature_trans
            if (feature == 'BxCxT_MntMeatProducts'):
                print(best_trans_label)
            self.best_bx_cx_dict[feature] = (best_trans_label, best_power_trans)
            # 3) 2) append transformed feature to the data frame
            self.training[feature] = best_power_trans
            # 3) 3) apply the best Box-Cox transformation, determined on training data, on unseen data
            self.unseen[feature] = np.round(self._bx_cx_trans_dict[best_trans_label](self.unseen[feature]), 4)
        self.box_cox_features = num_features_BxCx



    def rank_features_chi_square(self, continuous_flist, categorical_flist):
        """"
            Method to rank all features according to chi-square test for independence in relation to Response.
            All based solely on the training set.
        """
        chisq_dict = {}
        if continuous_flist:
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            for feature in continuous_flist:
                feature_bin = bindisc.fit_transform(self.training[feature].values[:, np.newaxis])
                feature_bin = pd.Series(feature_bin[:, 0], index=self.training.index)
                cont_tab = pd.crosstab(feature_bin, self.training["Response"], margins=False)
                chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]
        if categorical_flist:
            for feature in categorical_flist:
                cont_tab = pd.crosstab(self.training[feature], self.training["Response"], margins=False)
                chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]

        df_chisq_rank = pd.DataFrame(chisq_dict, index=["Chi-Squared", "p-value"]).transpose()
        df_chisq_rank.sort_values("Chi-Squared", ascending=False, inplace=True)
        df_chisq_rank["valid"] = df_chisq_rank["p-value"] <= 0.05
        self._rank["chisq"] = df_chisq_rank

    def calc_dta_feat_worth(self, feat_list, max_depth, min_samples_split, min_samples_leaf, seed):
        """"
            Method that receives a list of feature names, name of target and DecisionTreeClassifier paramethers and
            returns a df with all features with a worth higher than zero. All based solely on the training set.
        """

        # Preparing the Input Data for the DTA
        X = self.training.loc[:, feat_list].values
        y = self.training["Response"].values

        # Run the estimation through DecisionTreeClassifier
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf = min_samples_leaf, random_state=seed)
        # Fits the DTClassifier with our data
        dtree = dtree.fit(X, y)

        # Create a dictionary with the name of all features and its importance according to the DTA estimation
        fi = dict(zip(feat_list, dtree.feature_importances_))
        # Then creates a Dataframe with it
        fidf = pd.DataFrame(fi, index=["worth"])
        # Transpose it because the way it is created it is on the other orientation
        fidf_t = fidf.transpose().sort_values(by="worth", ascending=False)
        # Removes features with worth 0 and puts it into a df called worth_df
        worth_df = fidf_t[fidf_t.worth > 0]

        self._rank["dta"] = worth_df

    def print_top(self, n=10):
        """"
            Prints the best n features (default n = 10)
        """
        print(self._rank.index[0:n])

    def get_top(self, criteria, n_top):
        """"
            Returns the training and unseen datasets with only the best n_top features according to the criteria
            selected (chi_square or dta) (default n_top = 10).
        """
        input_features = list(self._rank[criteria].index[0:n_top])
        input_features.append("Response")
        return self.training[input_features], self.unseen[input_features]



    def _input_missing_values(self):
        """"
            Inputs any missing values of numerical features with its mean in order to deal with the weird missings.

            NOTE: we are doing this here due to missing values in the newly engineered features that could
            appear due to weird interaction between features, like dividing by zero or something of that nature.
        """

        def input_missing(df):
            num_feat_list = df._get_numeric_data().drop(["Response", "Education", "Marital_Status"], axis=1).columns
            for feat in num_feat_list:
                if (df[feat].isna().sum() > 0):
                    df[feat] = df[feat].fillna(df[feat].mean())

            return

        # We need to apply on training AND on unseen data, since the Box-Cox transformations were applied on both datsets.
        input_missing(self.training)
        input_missing(self.unseen)

        return