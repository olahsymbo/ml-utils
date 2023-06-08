import numpy as np


class DataImbalance:
    """
    :param data: input data with imbalanced samples
    :param label: corresponding labels
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def fit_oversample(self, min_label, ratio):
        """
        :param min_label: labels of the smaller class
        :param ratio: oversampling ratio
        :return: oversampled data and oversampled label
        """

        # get both majority and minority samples
        maj_samples = self.data[self.label != min_label]
        min_samples = self.data[self.label == min_label]

        # get the number of samples to be duplicated
        num_duplicate = int(ratio * len(min_samples))

        # duplicate minority class samples randomly
        duplicate_ind = np.random.choice(len(min_samples), num_duplicate, replace=True)
        duplicate_samples = min_samples[duplicate_ind]

        # merge both maj samples and duplicated min samples
        oversampled_data = np.concatenate([maj_samples, duplicate_samples])
        oversampled_label = np.concatenate([self.label[self.label != min_label],
                                            self.label[self.label == min_label]])

        return oversampled_data, oversampled_label

    def fit_undersample(self, maj_label, ratio):
        """
        :param maj_label: labels of the larger class
        :param ratio: undersampling ratio
        :return: undersampled data and undersampled labels
        """

        # get both majority and minority samples
        maj_samples = self.data[self.label == maj_label]
        min_samples = self.data[self.label != maj_label]

        # get the number of samples to be retained
        num_retain = int(ratio * len(maj_label))

        # duplicate minority class samples randomly
        retain_ind = np.random.choice(len(maj_samples), num_retain, replace=True)
        retained_samples = maj_samples[retain_ind]

        # merge both maj samples and duplicated min samples
        undersampled_data = np.concatenate([retained_samples, min_samples])
        undersampled_label = np.concatenate([self.label[self.label != maj_label],
                                            self.label[self.label == maj_label]])

        return undersampled_data, undersampled_label
