def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    nTruePositives = 0
    nFalsePositives = 0
    for i in len(prediction):
        if prediction[i] == True:
            if prediction[i] == ground_truth[i]:
                nTruePositives = nTruePositives + 1
            else:
                nFalsePositives = nFalsePositives + 1
        else :
            if prediction[i] == ground_truth[i]:
                nTrueNegatives = nTrueNegatives + 1
            else:
                 nFalseNegatives = nFalseNegatives + 1

    Recall = nTruePositives / (nTruePositives + nFalseNegatives)
# Recall = TruePositives / (TruePositives + FalseNegatives)
# TODO: implement metrics!
# Some helpful links:
# https://en.wikipedia.org/wiki/Precision_and_recall
# https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
