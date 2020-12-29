import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import models
import dataset
import numpy as np
from sklearn.metrics import recall_score, balanced_accuracy_score


def build_confusion_matrix(orig_lables, predict_class, title):
    """
    given labels it plots a confusion matrix
    :param orig_lables: ground truth
    :param predict_class: predictions
    :param title: title of the plot
    """
    con_met = confusion_matrix(orig_lables, predict_class)
    cmn = con_met.astype('float') / con_met.sum(axis=1)[:, np.newaxis]
    conDisplay = ConfusionMatrixDisplay(cmn, display_labels=["car", "truck", "cat"])
    conDisplay.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.show()


def create_gt_prediction(data, model):
    """
    creating 2 lists containing ground truth and model prediction
    :param data: data
    :param model: model
    :return: gt and prediction
    """
    gt = np.zeros(len(data))
    model_pred = np.zeros(len(data))
    for i in range(len(data)):
        item = data[i][0]
        pred = np.reshape(item, (1, *item.shape))
        prediction = np.argmax(model(pred).detach().numpy())
        gt[i] = data[i][1]
        model_pred[i] = prediction
    return gt, model_pred


def calculate_result(model, dev_path, train_path):
    """
    over 2 paths of train and dev sets, gt and prediction are returned
    :param model: model
    :param dev_path: path to dev file
    :param train_path: path to train file
    :return: gt and prediction for both data
    """
    test_data = dataset.get_dataset_as_torch_dataset(dev_path)
    train_data = dataset.get_dataset_as_torch_dataset(train_path)

    gt_train, prediction_train = create_gt_prediction(train_data, model)
    gt_test, prediction_test = create_gt_prediction(test_data, model)

    return gt_test, prediction_test, gt_train, prediction_train


if __name__ == '__main__':
    model = models.SimpleModel()
    model.load('./data/final_model.ckpt')
    gt_test, prediction_test, gt_train, prediction_train = calculate_result(model, './data/dev.pickle',
                                                                            './data/train.pickle')
    print(
        f'on train:\nbalanced accuracy: {balanced_accuracy_score(gt_train, prediction_train)}\nrecall: {recall_score(gt_train, prediction_train, average=None)}')
    print(
        f'on test:\nbalanced accuracy: {balanced_accuracy_score(gt_test, prediction_test)}\nrecall: {recall_score(gt_test, prediction_test, average=None)}')
    # gt_test, prediction_test, gt_train, prediction_train = calculate_result(model)
    build_confusion_matrix(gt_test, prediction_test, 'test set')
    build_confusion_matrix(gt_train, prediction_train, 'train set')
    # print("train accuracy is: %.3f" % ((np.count_nonzero(gt_train == prediction_train))/len(gt_train)))
    # print("test accuracy is: %.3f" % ((np.count_nonzero(gt_test == prediction_test))/len(gt_test)))
