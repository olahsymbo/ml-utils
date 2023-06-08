import numpy as np
from sklearn.linear_model import LinearRegression


class DataImputation:
    """
    :param data: the input data with missing values
    :param method: imputation method. Either "mean", "median", "mode" or "regression"

    :return imputed_data: output data with filled data points.
    """

    def __init__(self, data, method="mean"):

        self.data = data
        self.method = method

    def fit_imputation(self):
        imputed_data = self.data.copy()

        if self.method == "mean":
            for col in range(imputed_data.shape[1]):
                missing_indices = np.isnan(imputed_data[:, col])
                mean = np.mean(imputed_data[~missing_indices, col])
                imputed_data[missing_indices, col] = mean

        elif self.method == "median":
            for col in range(imputed_data.shape[1]):
                missing_indices = np.isnan(imputed_data[:, col])
                median = np.median(imputed_data[~missing_indices, col])
                imputed_data[missing_indices, col] = median

        elif self.method == "mode":
            for col in range(imputed_data.shape[1]):
                missing_indices = np.isnan(imputed_data[:, col])
                mode, counts = np.unique(imputed_data[~missing_indices, col], return_counts=True)
                mode_value = mode[np.argmax(counts)]
                imputed_data[missing_indices, col] = mode_value

        elif self.method == "regression":
            for col in range(imputed_data.shape[1]):
                missing_indices = np.isnan(imputed_data[:, col])
                observed_indices = ~missing_indices

                X = imputed_data[observed_indices, np.arange(imputed_data.shape[1]) != col]
                y = imputed_data[observed_indices, col]

                model = LinearRegression()
                model.fit(X, y)

                imputed_data[missing_indices, col] = model.predict(
                    imputed_data[missing_indices, np.arange(imputed_data.shape[1]) != col])

        return imputed_data


class DataFiltering:
    """
    :param data: the input data with missing values
    :param method: data removal/filtering method.  Either by "row" or "column"
    :param percentage: quantity of data to drop from row or column

    :return filtered_data: output data with dropped data points.
    """

    def __init__(self, data, percentage, method="row"):

        self.data = data
        self.method = method
        self.percentage = percentage

    def filter_data(self):
        filtered_data = self.data.copy()

        if self.method == "row":
            null_percentage = filtered_data.isnull().sum(axis=1) / filtered_data.shape[1] * 100
            rows_to_drop = null_percentage[null_percentage > self.percentage].index
            filtered_data.drop(rows_to_drop, inplace=True)

        elif self.method == "column":
            null_percentage = filtered_data.isnull().sum() / filtered_data.shape[0] * 100
            col_to_drop = null_percentage[null_percentage > self.percentage].keys()
            filtered_data.drop(col_to_drop, axis=1, inplace=True)

        return filtered_data
