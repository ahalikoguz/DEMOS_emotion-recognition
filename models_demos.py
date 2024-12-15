import torch
import torch.nn as nn
import torchvision

class slowfast_r50_Modelim(nn.Module):
    def __init__(self, num_classes):
        super(slowfast_r50_Modelim, self).__init__()
        self.num_classes = num_classes
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.model.blocks[6].proj = nn.Sequential(nn.Linear(2304, num_classes))

    def forward(self, x):
        x = self.model(x)
        return x


class r3d_18_Modelim(nn.Module):
    def __init__(self, num_classes):
        super(r3d_18_Modelim, self).__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.video.r3d_18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class x3d_Modelim_medium(nn.Module):
    def __init__(self, num_classes):
        super(x3d_Modelim_medium, self).__init__()
        self.num_classes = num_classes
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        self.model.blocks[5].proj = nn.Sequential(nn.Linear(2048, num_classes))

    def forward(self, x):
        return self.model(x)
