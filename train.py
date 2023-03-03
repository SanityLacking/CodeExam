import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from datasets_cifar10 import trainloader, validationloader, testloader

from model import Net


## load the model
modelName = "./models/model.pt"
net = torch.load(modelName)

NUM_EPOCHS = 10
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader,0),unit="batch",total=len(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        # loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        # loop.set_postfix(loss=loss, acc=)

        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
       

    # Evaluate the model on the test data
    net.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validationloader:
            output = net(data)
            validation_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validationloader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: ({:.0f}%)'.format(
        epoch, loss.item(), validation_loss, 100. * correct / len(validationloader.dataset)))
        

    torch.save(net,modelName)

print('Finished Training')
