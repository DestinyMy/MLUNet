import argparse
import copy
import sys

import torch
import os, random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import get_correlation

"""
MLP回归模型
网络输入备选： 初始通道数，每阶段深度，每阶段卷积深度，FLOPs，params，寒武纪核数
网络输出： 寒武纪延迟
"""
parser = argparse.ArgumentParser(description='predictor configurations')
parser.add_argument('-data_path', type=str, default='sample_dataset')
parser.add_argument('-pretrained', type=str, default='sample_dataset', help='path of mlp pretrained weights')
parser.add_argument('-seed', type=int, default=1000)
parser.add_argument('-rate', type=float, default=0.9)

parser.add_argument('-input_c', type=int, default=13)
parser.add_argument('-lr', type=float, default=8e-4)
parser.add_argument('-train_batch_size', type=int, default=2880)
parser.add_argument('-val_batch_size', type=int, default=320)
parser.add_argument('-epochs', type=int, default=2000)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def build_dataset():
    dataset = []
    with open(os.path.join(os.getcwd(), args.data_path, 'dataset.txt'), 'r') as f:
        while True:
            line = f.readline()
            if line:
                dataset.append(eval(line))
            else:
                break

    inputs = []
    targets = []
    for item in dataset:
        inputs.append(item[0])
        targets.append(item[1])
    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


class Net(nn.Module):
    def __init__(self, n_feature=10, n_layers=2, n_hidden=300, n_output=1, drop=0.2):
        super(Net, self).__init__()

        self.stem = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU())  # embedding

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        self.drop = nn.Dropout(p=drop)
        self.regression = nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regression(x)  # linear output
        x = x.squeeze(-1)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


class MLP:
    def __init__(self):
        self.model = Net(n_feature=args.input_c)

    def fit(self, inputs, targets):
        self.model = train(self.model, inputs, targets)
        self.save()

    def predict(self, test_data):
        return predict(self.model, test_data)

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(os.getcwd(), args.data_path, 'best_net.pth'))


def build_train_Optimizer_Loss(model):
    model.cuda()
    criterion = nn.SmoothL1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    return criterion, optimizer, scheduler


def train(model, inputs, targets):
    def train_one_epoch(model, train_inputs, train_targets, criterion, optimizer):
        model.train()
        optimizer.zero_grad()

        for i in range(0, len(train_inputs), args.train_batch_size):
            input = train_inputs[i:i+args.train_batch_size].cuda()
            target = train_targets[i:i+args.train_batch_size].cuda()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    slice = int(len(inputs)*args.rate)
    train_inputs = torch.from_numpy(inputs[:slice]).float()
    train_targets = torch.from_numpy(targets[:slice]).float()

    valid_inputs = torch.from_numpy(inputs[slice:]).float()
    valid_targets = torch.from_numpy(targets[slice:]).float()

    if args.pretrained is not None:
        print("Constructing MLP surrogate model with pre-trained weights")
        init = torch.load(os.path.join(os.getcwd(), args.pretrained, 'best_net.pth'), map_location='cpu')
        model.load_state_dict(init)
        best_net = copy.deepcopy(model)
    else:
        model = model.cuda()
        criterion, optimizer, scheduler = build_train_Optimizer_Loss(model)

        best_loss = 1e33
        for epoch in range(args.epochs):
            train_one_epoch(model, train_inputs, train_targets, criterion, optimizer)

            loss_val = validate(model, valid_inputs, valid_targets, criterion, args)
            scheduler.step()

            if loss_val < best_loss:
                best_loss = loss_val
                best_net = copy.deepcopy(model)

    # cal_correlation(best_net, valid_inputs, valid_targets)
    plot_fit(best_net, valid_inputs, valid_targets)

    return best_net.to('cpu')


def validate(model, valid_inputs, valid_targets, criterion):
    model.eval()

    loss = 0
    with torch.no_grad():
        for i in range(0, len(valid_inputs), args.val_batch_size):
            input = valid_inputs[i:i + args.val_batch_size].cuda()
            target = valid_targets[i:i + args.val_batch_size].cuda()
            output = model(input)
            loss_tmp = criterion(output, target)
            loss += loss_tmp.item()/(len(valid_inputs)*1.0/args.val_batch_size)

    return loss


def cal_correlation(model, inputs, targets):
    model.cuda()
    model.eval()

    with torch.no_grad():
        inputs, targets = inputs.cuda(), targets.cuda()
        pred = model(inputs)
        pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()

        rmse, rho, tau = get_correlation(pred, targets)

    #  dataset rmse: 0.3493722081184387, spearmanr: 0.9794119083898346, kendalltau: 0.8799636605189123
    print('rmse: {}, spearmanr: {}, kendalltau: {}'.format(rmse, rho, tau))
    return rmse, rho, tau, pred, targets


def predict(net, data_query):
    if data_query.ndim < 2:
        data = torch.zeros(1, data_query.shape[0])
        data[0, :] = torch.from_numpy(data_query).float()
    else:
        data = torch.from_numpy(data_query).float()

    net = net.cuda()
    net.eval()
    with torch.no_grad():
        data = data.cuda()
        pred = net(data)

    return pred.cpu().detach().numpy()


if __name__ == '__main__':
    inputs, targets = build_dataset()

    mlp_predictor = MLP()
    mlp_predictor.fit(inputs, targets)
