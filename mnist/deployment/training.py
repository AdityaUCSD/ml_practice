#IMPORTS
#######################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import bentoml

from MNIST_CNN import Network

# PARAMS
#######################################
n_epochs = 40
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.008
momentum = 0.5
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



def load_and_split_dataset(infile):
    f = gzip.open(infile, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    train_2d = np.array([im.reshape(-1, 28) for im in training_data[0]])
    test_2d = np.array([im.reshape(-1, 28) for im in test_data[0]])
    valid_2d = np.array([im.reshape(-1, 28) for im in validation_data[0]])

    training_data = TensorDataset(tensor(train_2d), tensor(training_data[1]))
    train_loader = DataLoader(training_data, batch_size=batch_size_train, shuffle=True)
    validation_data = TensorDataset(tensor(valid_2d), tensor(validation_data[1]))
    validation_loader = DataLoader(validation_data, batch_size=batch_size_test, shuffle=True)
    test_data = TensorDataset(tensor(test_2d), tensor(test_data[1]))
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    return train_loader, validation_loader, test_loader


def train(train_loader, network, epoch, train_counter, train_losses):
    
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.reshape(-1, 1, 28, 28)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), '../model/model.pth')
            torch.save(optimizer.state_dict(), '../model/optimizer.pth')
    return network, train_counter, train_losses

def test(validation_loader, network, test_counter, test_losses):
    
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.reshape(-1, 1, 28, 28)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(validation_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {test_loss}, Accuracy: {correct}/{len(validation_loader.dataset)} ({100. * correct / len(validation_loader.dataset)}%)\n')
    return test_counter, test_losses


def plot_losses(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('../figures/losses.png')

def plot_examples(test_loader, network):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    ex = example_data.reshape(-1, 1, 28, 28)


    with torch.no_grad():
        output = network(ex)
    fig = plt.figure(figsize=(14, 6))
    for i in range(10):
        plt.subplot(2,5,i+1)
        #plt.tight_layout()
        plt.imshow(example_data[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {} Truth: {}".format(output.data.max(1, keepdim=True)[1][i].item(), example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.savefig('../figures/examples.png')

if __name__ == "__main__":
    data_file = '../../data/mnist.pkl.gz'
    train_loader, validation_loader, test_loader = load_and_split_dataset(data_file)

    train_counter, train_losses = [], []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs+1)]
    test_losses = []

    network = Network()
    test_counter, test_losses = test(validation_loader, network, test_counter, test_losses)
    for epoch in range(1, n_epochs + 1):
        network, train_counter, train_losses = train(train_loader, network, epoch, train_counter, train_losses)
        test_counter, test_losses = test(validation_loader, network, test_counter, test_losses)

    plot_losses(train_counter, train_losses, test_counter, test_losses)

    plot_examples(test_loader, network)
