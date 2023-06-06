def calculate_f1_score(tp, fp, fn):

    """
    :param tp: true positives
    :param fp: false positives
    :param fn: false negatives
    :return: f1 score
    """

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score
