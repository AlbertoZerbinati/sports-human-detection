# Alberto Zerbinati

from torchvision import models
import torch.nn as nn

# Constants for architecture dimensions and dropout rate
OUT_RESNET = 512
SIZE_MLP = 200
RATE_DROPOUT = 0.3

class PeopleDetectionCNN(nn.Module):
    def __init__(self, dev):
        super(PeopleDetectionCNN, self).__init__()
        self.dev = dev  # Device (CPU or GPU)

        # Initialize pre-trained ResNet model
        resnet_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Remove the last fully-connected layer to use ResNet as a feature extractor
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-1])
        
        # Move ResNet to the specified device
        self.resnet = self.resnet.to(dev)

        # Freeze all layers in ResNet except the last one
        for p in self.resnet.parameters():
            p.requires_grad = False
        for p in self.resnet[7].parameters():
            p.requires_grad = True

        # Define a two-layer MLP with dropout and activation functions
        self.layer1 = nn.Linear(OUT_RESNET, SIZE_MLP).to(dev)
        self.act = nn.Tanh().to(dev)
        self.layer2 = nn.Linear(SIZE_MLP, 2).to(dev)
        self.final_act = nn.Softmax(dim=-1).to(dev)
        
        # Add dropout layer
        self.drop = nn.Dropout(RATE_DROPOUT).to(dev)

    def forward(self, inp):
        """Forward pass of the neural network."""
        # Move input to the specified device
        inp = inp.to(self.dev)
        
        # Feature extraction using ResNet
        feat = self.resnet(inp).squeeze()
        
        # Move features to the specified device
        feat = feat.to(self.dev)

        # Apply MLP to features and get output
        output = self.apply_mlp(feat)

        return output

    def apply_mlp(self, inp_feat):
        """Apply the MLP layers to the input features."""
        return self.final_act(
            self.layer2(self.drop(self.act(self.layer1(self.drop(inp_feat)))))
        )
