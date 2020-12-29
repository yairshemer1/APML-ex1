import models
from dataset import *
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr
import torch.nn as nn
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import evaluate_model as ev
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, balanced_accuracy_score


def class_imbalance_sampler(labels):
    """
    create a weighted sampler
    :param labels: labels of the data
    :return: sampler
    """
    class_count = torch.bincount(labels.squeeze()).type('torch.DoubleTensor')
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    len(labels),
                                    replacement=True)
    return sampler


def create_labels(trainset):
    """
    create labels of the data
    :param trainset: data
    :return: the labels
    """
    labels = []
    for item in trainset:
        labels.append(item[1])
    return labels


def train_model(model, data_loader, epochs, lr):
    """
    train model and return all possible information regarding the model's results
    :param model: model
    :param data_loader: data
    :param epochs: number of epochs
    :param lr: learn rate
    :return: model's results
    """
    loss_list = []
    test_accuracy = []
    train_accuracy = []
    car_accuracy_test = []
    truck_accuracy_test = []
    cat_accuracy_test = []
    car_accuracy_train = []
    truck_accuracy_train = []
    cat_accuracy_train = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            l2_regularization = torch.tensor(0.0)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2) ** 2

            # with reg term
            # loss = criterion(outputs, labels) + 0.01 * torch.sqrt(l2_regularization)
            # without reg term
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        gt_test, prediction_test, gt_train, prediction_train = ev.calculate_result(model, './data/clean_dev.pickle',
                                                                                   './data/clean_train.pickle')
        loss_list.append(running_loss / len(data_loader))
        train_accuracy.append(balanced_accuracy_score(gt_train, prediction_train))
        test_accuracy.append(balanced_accuracy_score(gt_test, prediction_test))
        acc_test = recall_score(gt_test, prediction_test, [0, 1, 2], average=None)
        acc_train = recall_score(gt_train, prediction_train, [0, 1, 2], average=None)
        car_accuracy_test.append(acc_test[0])
        truck_accuracy_test.append(acc_test[1])
        cat_accuracy_test.append(acc_test[2])
        car_accuracy_train.append(acc_train[0])
        truck_accuracy_train.append(acc_train[1])
        cat_accuracy_train.append(acc_train[2])
        print('epoch is: %d loss is: %.3f train acc: %.3f test acc: %.3f' % (
            epoch + 1, running_loss / len(data_loader), train_accuracy[-1], test_accuracy[-1]))

    print('Finished Training')
    return model, loss_list, test_accuracy, train_accuracy, car_accuracy_test, truck_accuracy_test, cat_accuracy_test, \
           car_accuracy_train, truck_accuracy_train, cat_accuracy_train


def clean_data(model, path, new_path, threshold):
    """
    a function to create a new data that is cleaned of bad labeling
    :param model: simple model to check loss on
    :param path: path to data
    :param new_path: path to save the new clean data
    :param threshold: threshold to delete loss higher then that
    :return: saves a pickle
    """
    data_set = get_dataset_as_torch_dataset(path)
    dev_loader = DataLoader(data_set, batch_size=1)
    criterion = nn.CrossEntropyLoss()
    new_data = []
    # losses=[]
    # j = 0
    for data in dev_loader:
        input, label = data
        if label == 0:
            outputs = model(input)
            # pred_label = np.argmax(outputs.detach().numpy())
            loss = float(criterion(outputs, label))
            # losses.append(loss)
            if loss > threshold:
                # plt.imsave(f'./images/loss {loss}_{j}.png', un_normalize_image(input[0]))
                # j += 1
                continue
        new_data.append((input[0], label[0]))
    # plt.plot(sorted(losses))
    # plt.xlabel('data')
    # plt.ylabel('loss')
    # plt.title('loss as a function of data - dev set\nsorted')
    # plt.show()
    new_data = MyDataset(new_data)
    # save new data
    with open(new_path, 'wb') as handle:
        pickle.dump(new_data, handle)


if __name__ == '__main__':
    """
    All stages of my work are mentioned here. Might look a bit messy but it is due to the different
    stages of my work that I had to keep. I tried to make it as informative as possible with comments
    where each stage start. Sorry for the mess.
    """
    model = models.SimpleModel()
    # train basic model on dirty data - 5 epochs and weighted sampler
    # trainset = get_dataset_as_torch_dataset('./data/train.pickle')
    # labels = torch.LongTensor(create_labels(trainset))
    # sampler = class_imbalance_sampler(labels)
    #
    # train_loader = DataLoader(trainset,
    #                           batch_size=4,
    #                           num_workers=2,
    #                           sampler=sampler)
    # epochs = 5
    # model, loss_list, test_acc, train_acc = train_model(model, train_loader, epochs)
    # PATH = './data/simple_model.ckpt'
    # model.save(PATH)

    # clean data by loading './data/simple_model.ckpt'
    # model.load('./data/simple_model.ckpt')
    # clean_data(model, './data/train.pickle', './data/clean_train.pickle', 1.5)
    # clean_data(model, './data/dev.pickle', './data/clean_dev.pickle', 1.5)

    # train model on clean data
    trainset = get_dataset_as_torch_dataset('./data/clean_train.pickle')
    labels = torch.LongTensor(create_labels(trainset))
    sampler = class_imbalance_sampler(labels)
    lrs = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    train_loader = DataLoader(trainset,
                              batch_size=4,
                              num_workers=2,
                              sampler=sampler)
    epochs = 20
    transforms = tr.Compose(
        [tr.ToPILImage(),
         tr.Resize((45, 45)),
         tr.RandomApply([
             tr.RandomAffine(degrees=30, scale=(0.8, 1.2)),
             tr.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
         ], 0.7),
         tr.RandomHorizontalFlip(),
         tr.RandomResizedCrop((32, 32), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
         tr.ToTensor()
         ])
    trainset.transform = transforms
    # acc_test_lr = []
    # acc_train_lr = []
    # for lr in lrs:
    #     model, loss_list, test_acc, train_acc, car_acc_test, truck_acc_test, cat_acc_test, \
    #     car_acc_train, truck_acc_train, cat_acc_train = train_model(model, train_loader, epochs, lr)
    #     gt_test, prediction_test, gt_train, prediction_train = ev.calculate_result(model, './data/clean_dev.pickle',
    #                                                                                './data/clean_train.pickle')
    #     acc_test_lr.append(test_acc)
    #     acc_train_lr.append(train_acc)
    #     PATH = f'./data/final_model_lr={lr}.ckpt'
    #     model.save(PATH)
    # # lr graph
    # for i in range(len(lrs)):
    #     plt.plot(acc_test_lr[i])
    # plt.legend(lrs)
    # plt.title('test set - each plot is an accuracy with a specific lr')
    # plt.show()
    # for i in range(len(lrs)):
    #     plt.plot(acc_train_lr[i])
    # plt.legend(lrs)
    # plt.title('train set - each plot is an accuracy with a specific lr')
    # plt.show()
    # create graphs for a model
    # plt.plot(loss_list)
    # plt.plot(test_acc)
    # plt.plot(train_acc)
    # plt.title(
    #     'loss value, train and dev accuracy as a function of epochs\nbatch_size = 4\nwithout augmentation\nwith weighted sampler\nwith regularization term')
    # plt.legend(['loss', 'test accuracy', 'train accuracy'])
    # plt.show()
    # plt.plot(car_acc_test)
    # plt.plot(cat_acc_test)
    # plt.plot(truck_acc_test)
    # plt.title(
    #     'accuracy over the test set on each class as a function of epochs\nbatch_size = 4\nwithout augmentation\nwith weighted sampler\nwith regularization term')
    # plt.legend(['car', 'cat', 'truck'])
    # plt.show()
    # plt.plot(car_acc_train)
    # plt.plot(cat_acc_train)
    # plt.plot(truck_acc_train)
    # plt.title(
    #     'accuracy over the train set on each class as a function of epochs\nbatch_size = 4\nwithout augmentation\nwith weighted sampler\nwith regularization term')
    # plt.legend(['car', 'cat', 'truck'])
    # plt.show()
