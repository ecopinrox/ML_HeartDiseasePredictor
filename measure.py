import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def getConfusionMatrixEntries(predicted: nparray, actual: nparray):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(predicted.size):
        if(predicted[i] == 1):
            if(actual[i] == 1):
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if(actual[i] == 1):
                fn = fn + 1
            else:
                tn = tn + 1

    return tp, fp, fn, tn


def accuracy(tp: int, fp: int, fn: int, tn: int):
    return (tp + tn) / (tp + fp + fn + tn)


def precision(tp: int, fp: int, fn: int, tn: int):
    return tp / (tp + fp)


def recall(tp: int, fp: int, fn: int, tn: int):
    return tp / (tp + fn)


def specificity(tp: int, fp: int, fn: int, tn: int):
    return tn / (tn + fp)


def f1score(tp: int, fp: int, fn: int, tn: int):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return (2 * p * r) / (p + r)


def confusionMatrix(tp: int, fp: int, fn: int, tn: int):
    values = pd.DataFrame(data={"Actual positive":[tp, fp], "Actual negative":[fn, tn]}, index=["Predicted positive", "Predicted negative"])
    plt.title("Confusion matrix")
    sns.heatmap(values, annot=True, cmap='Blues')
    plt.show()
    print("Accuracy:", round(accuracy(tp, fp, fn, tn), 2))
    print("Precision:", round(precision(tp, fp, fn, tn), 2))
    print("Recall:", round(recall(tp, fp, fn, tn), 2))
    print("Specificity:", round(specificity(tp, fp, fn, tn), 2))
    print("F1 score:", round(f1score(tp, fp, fn, tn), 2))
    
    