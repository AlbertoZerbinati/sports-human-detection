# Alberto Zerbinati

import torch
import torch.nn as nn
import torchvision.models as models

RESNET_OUT = 512
MLP_SIZE = 200
DROPOUT = 0.3


class PeopleDetectionCNN(nn.Module):
    def __init__(self, device):
        super(PeopleDetectionCNN, self).__init__()
        self.device = device

        # Adding resnet
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Removing last fully connected layer
        self.resnet = torch.nn.Sequential(*(list(resnet34.children())[:-1]))
        self.resnet = self.resnet.to(device)
        # Freezing all params but the last layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet[7].parameters():
            param.requires_grad = True

        # initialize MLP feedforward
        self.w1 = torch.nn.Linear(RESNET_OUT, MLP_SIZE, bias=True).to(device)
        self.activation = torch.nn.Tanh().to(device)
        self.w2 = torch.nn.Linear(MLP_SIZE, 2, bias=True).to(device)
        self.softmax = torch.nn.Softmax(dim=-1).to(device)

        # Adding droput for regularization
        self.dropout = torch.nn.Dropout(DROPOUT).to(device)

    def forward(self, x):
        x = x.to(self.device)

        # Obtaining the features of the image
        h = self.resnet(x).squeeze()
        h = h.to(self.device)

        # Using the MLP on features
        out = self.mlp(h)

        return out

    def mlp(self, x):
        return self.softmax(
            self.w2(self.dropout(self.activation(self.w1(self.dropout(x)))))
        )
