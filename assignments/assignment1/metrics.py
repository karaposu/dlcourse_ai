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
    nTrueNegatives = 0
    nFalseNegatives = 0
    # print("len prediction", len(prediction))
    # print("prediction", prediction)
    # print("type prediction", type(prediction))

    for i in range(len(prediction)):
        if prediction[i] == True:
            if prediction[i] == ground_truth[i]:
                nTruePositives = nTruePositives + 1
            else:
                nFalsePositives = nFalsePositives + 1
        else:
            if prediction[i] == ground_truth[i]:
                nTrueNegatives = nTrueNegatives + 1
            else:
                nFalseNegatives = nFalseNegatives + 1

    recall = nTruePositives / (nTruePositives + nFalseNegatives)
    precision = nTruePositives/ (nTruePositives + nFalsePositives)
    accuracy =  (nTruePositives + nTrueNegatives) / (nTruePositives + nTrueNegatives + nFalseNegatives+ nFalsePositives)
    # print("nTruePositives:", nTruePositives)
    # print("nTrueNegatives:", nTrueNegatives)
    # print("nFalseNegatives:", nFalseNegatives)
    # print("nFalsePositives:", nFalsePositives)

    f1 = (2 *precision *recall) / (precision + recall)
    #
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

    accuracy = 0


    nTruePositives = 0
    nFalsePositives = 0
    nTrueNegatives = 0
    nFalseNegatives = 0
    # print("len prediction", len(prediction))
    # print("prediction", prediction)

    for i in range(len(prediction)):

        if prediction[i] == ground_truth[i]:
            nTruePositives = nTruePositives + 1
        else:
            nFalsePositives = nFalsePositives + 1



    accuracy = nTruePositives  / (nTruePositives  +nFalsePositives)
    return  accuracy
