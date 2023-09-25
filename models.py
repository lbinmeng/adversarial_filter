import torch
import torch.nn as nn
import numpy as np
import torchaudio.functional as F
from typing import Optional
from scipy.signal import butter


def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    x = x.reshape(x.size(0), -1)
    return x.shape[-1]


def LoadModel(model_name, n_classes, Chans, Samples):
    if model_name == 'EEGNet':
        model = EEGNet(n_classes=n_classes,
                       Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25)
    elif model_name == 'DeepCNN':
        model = DeepConvNet(n_classes=n_classes,
                            Chans=Chans, 
                            Samples=Samples, 
                            dropoutRate=0.5)
    else:
        raise 'No such model'
    return model


class EEGNet(nn.Module):
    """
    :param
    """
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 
                          0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))
        
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class DeepConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=25), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=50), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=100), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=CalculateOutSize([self.block1, self.block2, self.block3],
                                                    self.Chans, self.Samples),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5,
    ):
        super(ShallowConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=40,
                      out_channels=40,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=40), 
            Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            Activation('log'), 
            nn.Dropout(self.dropoutRate))
        
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=CalculateOutSize([self.block1],
                                                    self.Chans, self.Samples),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class FilterLayer(nn.Module):
    def __init__(self, order, band, fs):
        super(FilterLayer, self).__init__()

        B, A = butter(order, np.array(band) / (fs / 2), btype='bandpass')
        B = torch.from_numpy(B).type(torch.FloatTensor)
        A = torch.from_numpy(A).type(torch.FloatTensor)
        self.B = nn.Parameter(B, requires_grad=True)
        self.A = nn.Parameter(A, requires_grad=True)
    
    def forward(self, x):
        return F.lfilter(x, self.A, self.B)
    

class SpatialFilterLayer(nn.Module):
    def __init__(self, Channels):
        super(SpatialFilterLayer, self).__init__()

        self.Channels = Channels
        self.filter = nn.Parameter(torch.zeros(size=(self.Channels, self.Channels)))
    
    def forward(self, x):
        return torch.matmul(self.filter + torch.eye(self.Channels).to(x.device), x)
    