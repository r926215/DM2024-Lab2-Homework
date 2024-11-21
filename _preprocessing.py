import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

import os
from helpers import draw_statistic_graph as dsg
from helpers import data_mining_helpers as dmh
import _const as const

class Outlier():
    class Detect():
        def __init__(self):
            pass
        def IQR(self, data):

            features_for_IQR = data.columns.difference(const.exclude_features_for_Outliers)

            Q1 = data[features_for_IQR].quantile(0.25)
            Q3 = data[features_for_IQR].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_condition = (data[features_for_IQR] < lower_bound) | (data[features_for_IQR] > upper_bound)
            outliers = data[outlier_condition.any(axis=1)].copy()

            outliers['Outlier_Features'] = outlier_condition.apply(lambda x: ','.join(x.index[x]), axis=1)

            outliers.to_csv(os.path.join(const.dir_path["detect_outliers_dir"], 'detect_IQR_Method.csv'), index=False)

            feature_counts = outliers['Outlier_Features'].str.split(',', expand=True).stack().reset_index(drop=True)
            feature_occurrences = feature_counts.value_counts()

            self.Print_Info(outliers, feature_occurrences)

        def Z_Scores(self, data, threshold=3):

            features_for_Z_Scores = data.columns.difference(const.exclude_features_for_Outliers)

            # Calculate Z-Scores
            z_scores = (data[features_for_Z_Scores] - data[features_for_Z_Scores].mean()) / (data[features_for_Z_Scores].std())

            # Identify outliers
            outlier_condition = (abs(z_scores) > threshold)
            outliers = data[outlier_condition.any(axis=1)].copy()


            outliers['Outlier_Features'] = outlier_condition.apply(lambda x: ','.join(x.index[x]), axis=1)

            outliers.to_csv(os.path.join(const.dir_path["detect_outliers_dir"], 'detect_Zscores_Method.csv'), index=False)


            feature_counts = outliers['Outlier_Features'].str.split(',', expand=True).stack().reset_index(drop=True)
            feature_occurrences = feature_counts.value_counts()

            self.Print_Info(outliers, feature_occurrences)

        def Print_Info(self, outliers, feature_occurrences):

            print("The number of outliers:", len(outliers))

            dsg.Features_Occurrences(feature_occurrences)

    class Remove():
        def __init__(self):
            pass

        def IQR(self, data):

            features_for_IQR = data.columns.difference(const.exclude_features_for_Outliers)


            Q1 = data[features_for_IQR].quantile(0.25)
            Q3 = data[features_for_IQR].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_condition = (data[features_for_IQR] < lower_bound) | (data[features_for_IQR] > upper_bound)

            result_data = data[~(outlier_condition).any(axis=1)]

            result_data.to_csv(os.path.join(const.dir_path["remove_outliers_dir"], 'remove_IQR_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Z_Scores(self, data, threshold=3):

            features_for_Z_Scores = data.columns.difference(const.exclude_features_for_Outliers)

            # Calculate Z-Scores
            z_scores = (data[features_for_Z_Scores] - data[features_for_Z_Scores].mean()) / (data[features_for_Z_Scores].std())

            # Identify outliers
            outlier_condition = (abs(z_scores) > threshold)

            result_data = data[~outlier_condition.any(axis=1)]

            result_data.to_csv(os.path.join(const.dir_path["remove_outliers_dir"], 'remove_Z-Scores_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Unchange(self, data):

            result_data = data
            result_data.to_csv(os.path.join(const.dir_path["remove_outliers_dir"], 'not_change.csv'), index=False)
            return result_data
        
        def Print_Info(self, before_data, current_data):

            print("Remaining Data Length", len(current_data))

            dsg.Feature_Values_Distribution(before_data, current_data)

class MissingValues():
    def __init__(self):
        pass
    class Remove():
        def Remove_all(self, data):
            # Count by rows
            missing_count = data.isnull().apply(lambda x: dmh.check_missing_values(x), axis=1)

            # Reserve iff missing_count is 0
            result_data = data[missing_count == 0]

            result_data.to_csv(os.path.join(const.dir_path["remove_missing_value_dir"], 'by_missing_count_0.csv'), index=False)

            self.Print_info(data, result_data)

            return result_data

        def Remove_n(self, data, n=10):
            # Count by rows
            missing_count = data.isnull().apply(lambda x: dmh.check_missing_values(x), axis=1)

            # Reserve iff missing_count is less n 
            result_data = data[missing_count < n]

            result_data.to_csv(os.path.join(const.dir_path["remove_missing_value_dir"], f'by_missing_count_{n}.csv'), index=False)

            self.Print_info(data, result_data)

            return result_data

        def Unchange(self, data):

            result_data = data

            result_data.to_csv(os.path.join(const.dir_path["remove_missing_value_dir"], 'not_change.csv'), index=False)

            self.Print_info(data, result_data)

            return result_data

        def Print_info(self, before_data, current_data):

            print("Remaining Data Length:", len(current_data))
            dsg.Compare_Features_Occurrences(before_data, current_data)
            

    class Fill():
        def Fill_Mean(self, data):

            result_data = data.fillna(data.mean(), inplace=False)

            result_data.to_csv(os.path.join(const.dir_path["fill_missing_value_dir"], 'Mean_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Fill_Median(self, data): 

            result_data = data.fillna(data.median(), inplace=False)

            result_data.to_csv(os.path.join(const.dir_path["fill_missing_value_dir"], 'Median_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Fill_Forward(self, data):

            result_data = data.fillna(data.ffill(), inplace=False)

            result_data.to_csv(os.path.join(const.dir_path["fill_missing_value_dir"], 'Forward_Fill_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Fill_Backward(self, data):

            result_data = data.fillna(data.bfill(), inplace=False)

            result_data.to_csv(os.path.join(const.dir_path["fill_missing_value_dir"], 'Backward_Fill_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Fill_Interpolation(self, data, method='linear'):
            """
                method:
                    polynomial,
                    linear,
                    quadratic
            """
            result_data =  data.interpolate(method=method)

            result_data.to_csv(os.path.join(const.dir_path["fill_missing_value_dir"], 'Interpolation_Method.csv'), index=False)

            self.Print_Info(data, result_data)

            return result_data

        def Print_Info(self, before_data, current_data):

            dsg.Feature_Values_Distribution(before_data, current_data)

class Normalization():

    def __init__(self):
        pass

    def StandardScaler(self, data):
        scaler = StandardScaler()
        normalized_continuous = scaler.fit_transform(data)

        return normalized_continuous, scaler

    def MinMaxScaler(self, data):
        scaler = MinMaxScaler()
        normalized_continuous = scaler.fit_transform(data)

        return normalized_continuous, scaler

    def MaxAbsScaler(self, data):
        scaler = MaxAbsScaler()
        normalized_continuous = scaler.fit_transform(data)

        return normalized_continuous, scaler

    def RobustScaler(self, data):
        scaler = RobustScaler()
        normalized_continuous = scaler.fit_transform(data)

        return normalized_continuous, scaler

    def OneHotEncoder(self, data):
        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(data)

        return encoded_categorical, encoder
