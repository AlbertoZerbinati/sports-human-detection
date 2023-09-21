# Alberto Zerbinati

from torchvision import models
import torch.nn as nn

OUT_RESNET = 512
SIZE_MLP = 200
RATE_DROPOUT = 0.3

class PeopleDetectionCNN(nn.Module):
    def __init__(self, dev):
        super(PeopleDetectionCNN, self).__init__()
        self.dev = dev

        # Load resnet
        resnet_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Omit last FC layer
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-1])
        self.resnet = self.resnet.to(dev)
        
        # Freeze all but last layer params
        for p in self.resnet.parameters():
            p.requires_grad = False
        for p in self.resnet[7].parameters():
            p.requires_grad = True

        # Create feed-forward MLP
        self.layer1 = nn.Linear(OUT_RESNET, SIZE_MLP).to(dev)
        self.act = nn.Tanh().to(dev)
        self.layer2 = nn.Linear(SIZE_MLP, 2).to(dev)
        self.final_act = nn.Softmax(dim=-1).to(dev)

        # Add dropout
        self.drop = nn.Dropout(RATE_DROPOUT).to(dev)

    def forward(self, inp):
        inp = inp.to(self.dev)
        
        # Get image features
        feat = self.resnet(inp).squeeze()
        feat = feat.to(self.dev)

        # Apply MLP
        output = self.apply_mlp(feat)

        return output

    def apply_mlp(self, inp_feat):
        return self.final_act(
            self.layer2(self.drop(self.act(self.layer1(self.drop(inp_feat)))))
        )
