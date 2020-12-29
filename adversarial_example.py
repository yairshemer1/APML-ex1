from models import *
from dataset import *
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


def get_adversarial_examples(model, train_data, class_num):
    """
    create a list of indices for images labels 'class_num' of which the model predicted with high accuracy
    :param model: model
    :param train_data: data
    :param class_num: label
    :return: list of indices
    """
    indices = []
    for i in range(len(train_data)):
        im, label = train_data[i]
        outputs = model(torch.reshape(im, (1, *im.shape)))
        predictions = outputs.detach().numpy()[0]
        accuracy_per_class = np.exp(predictions) / np.sum(np.exp(predictions))
        if accuracy_per_class[class_num] > 0.99:
            indices.append(i)

    return indices


def train_adversarial(model, image, epochs, adv_label):
    """
    given a model and an image, the noise is learned so that the model will mistake the image's label
    after adding the learned noise to the image with a certain coefficient (0.007)
    :param model: model
    :param image: image as tensor
    :param epochs: number of epochs
    :param adv_label: the wrong label to mislead the model
    :return: the noise
    """
    noise = torch.randn((3, 32, 32), requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([noise],
                          lr=25,
                          momentum=0.9)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(torch.stack([image + 0.007 * noise]))

        loss = criterion(outputs, torch.tensor([adv_label]))
        loss.backward()
        optimizer.step()
    return noise


if __name__ == '__main__':
    dic = label_names()
    model = SimpleModel()
    model.load('./data/final_model_1.ckpt')
    train_data = get_dataset_as_array('./data/clean_train.pickle')
    indices = get_adversarial_examples(model, train_data, 0)
    image = train_data[indices[2]][0]
    # get noise
    noise = train_adversarial(model, image, 200, 2)
    # collect information
    real_label_output = model(torch.stack([image]))
    new_label_output = model(torch.stack([image + 0.007 * noise]))
    real_predictions = real_label_output.detach().numpy()[0]
    new_predictions = new_label_output.detach().numpy()[0]
    # soft max
    real_accuracy_per_class = np.exp(real_predictions) / np.sum(np.exp(real_predictions))
    real_label = np.argmax(real_accuracy_per_class)
    new_accuracy_per_class = np.exp(new_predictions) / np.sum(np.exp(new_predictions))
    new_label = np.argmax(new_accuracy_per_class)
    print(f'real label was {real_label} with accuracy {real_accuracy_per_class[real_label]},\n'
          f'new label is {new_label} with accuracy {new_accuracy_per_class[new_label]}')
    # draw results
    plt.imshow(un_normalize_image(image))
    plt.title(f'original image\nlabeled {dic[real_label]} with {float("{:.2f}".format(real_accuracy_per_class[real_label]*100))}%')
    plt.show()
    plt.imshow(un_normalize_image(image+0.007*noise.detach()))
    plt.title(f'noised image\nlabeled {dic[new_label]} with {float("{:.2f}".format(new_accuracy_per_class[new_label]*100))}%')
    plt.show()
    plt.imshow(un_normalize_image(noise.detach()))
    plt.title('noise added')
    plt.show()