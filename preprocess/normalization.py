import math


class Normalization:

    def __init__(self, data):
        self.data = data

    def min_max(self):
        min_val = self.data.min().min()
        max_val = self.data.max().max()
        normalized_df = (self.data - min_val) / (max_val - min_val)
        return normalized_df

    def z_score(self):
        mean = self.data.mean().mean()
        std_dev = self.data.std().std()
        normalized_df = (self.data - mean) / std_dev
        return normalized_df

    def decimal_scaling(self):
        max_abs = self.data.abs().max().max()
        power = math.ceil(math.log10(max_abs))
        normalized_df = self.data / (10 ** power)
        return normalized_df
