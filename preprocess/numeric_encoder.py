from sklearn.preprocessing import LabelEncoder


class NumericEncoder:

    def __init__(self, data):
        self.data = data

    def convert_to_numeric(self):
        data_new = self.data
        categorical_feature_mask = data_new.dtypes != int
        categorical_cols = data_new.columns[categorical_feature_mask].tolist()
        data_new[categorical_cols] = data_new[categorical_cols].apply(lambda col: LabelEncoder().fit_transform(col))
        return data_new
