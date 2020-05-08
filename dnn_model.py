import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.model_selection import train_test_split
import copy

gpu = torch.cuda.is_available()

class Net(nn.Module):
  def __init__(self, input_size):
    super(Net, self).__init__()
    
    # fully connected layers
    self.fc1 = nn.Linear(input_size, 30) 
    self.final_linear = nn.Linear(30, 1)

    # activation
    self.relu = nn.ReLU()

    # dropout
    self.dropout = nn.Dropout()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.final_linear(x)
    return x

def eval_model(model, loss_metric, data_loader):
    model.eval()
    
    total_loss = 0
    total_accuracy = 0
    total = 0
    for i, (X,Y) in enumerate(data_loader):
        if gpu:
          X, Y = X.cuda(), Y.cuda()
            
        P = model(X)
        loss = loss_metric(P, Y.float().unsqueeze(1))

        total += X.size(0)
        total_loss += loss.item()
        predictions = torch.round(torch.sigmoid(P))
        total_accuracy += (predictions.flatten().long() == Y.flatten().long()).sum().item()
    
    return total_loss / total, total_accuracy / total

def train_model(model, optimizer, loss_metric, train_loader, test_loader, epochs=5):
    model.train()

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_test_accuracy = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        
        total_train_loss = 0
        total_train_accuracy = 0
        total = 0
        
        for i, (X,Y) in enumerate(train_loader):
            if gpu:
              X, Y = X.cuda(), Y.cuda()
            
            optimizer.zero_grad()   
            
            P = model(X)
            loss = loss_metric(P, Y.float().unsqueeze(1))

            loss.backward()
            optimizer.step()
            
            total += X.size(0)
            total_train_loss += loss.item()

            predictions = torch.round(torch.sigmoid(P))
            total_train_accuracy += (predictions.flatten().long() == Y.flatten().long()).sum().item()

            if not total % 100:
                pass
                # print(total_train_loss / total, total_train_accuracy / total)
        
        train_losses.append(total_train_loss / total)
        train_accuracies.append(total_train_accuracy / total)
        
        test_loss, test_accuracy = eval_model(model, loss_metric, test_loader)

        if test_accuracy > best_test_accuracy:
          best_model = copy.deepcopy(model)
          best_test_accuracy = test_accuracy

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # print('End of Epoch %d: Training loss: %f, Training accuracy: %f, Testing loss: %f, Testing accuracy: %f' %
        #       (epoch, train_losses[-1], train_accuracies[-1], test_losses[-1], test_accuracies[-1]))
        
    return train_losses, train_accuracies, test_losses, test_accuracies, best_model, best_test_accuracy

def fit(model, train_x, train_y, batch=64, lr=0.001, epochs=100):
    model = copy.deepcopy(model)

    train_x = torch.Tensor(train_x).float()
    train_y = torch.Tensor(train_y)

    train_x, val_x, train_y, val_y = \
                train_test_split(train_x, train_y, test_size=0.1)

    train_dataset = data.TensorDataset(train_x, train_y)
    val_dataset  = data.TensorDataset(val_x, val_y)

    train_loader = data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader  = data.DataLoader(val_dataset, batch_size=batch, shuffle=False)

    if gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_metric = nn.BCEWithLogitsLoss()

    train_losses, train_accuracies, test_losses, test_accuracies, best_model, best_test_accuracy = \
        train_model(model, optimizer, loss_metric, train_loader, val_loader, epochs=epochs)
    
    _, acc = eval_model(best_model, loss_metric, val_loader)

    return best_model

def predict(model, test_x, test_y):
    test_x = torch.Tensor(test_x).float()
    test_y = torch.Tensor(test_y).flatten().long()
    
    P = model(test_x)

    predictions = torch.round(torch.sigmoid(P)).flatten().long()
    total_accuracy = (predictions == test_y).sum().item()

    return total_accuracy / test_x.size(0), predictions

def predict_no_eval(model, test_x):
    test_x = torch.Tensor(test_x).float()
    
    P = model(test_x)

    predictions = torch.round(torch.sigmoid(P)).flatten().long()

    return predictions

def save_model(model, output_file):
    torch.save(model.state_dict(), output_file)