import torch
import torchvision
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = torchvision.models.resnext50_32x4d(pretrained=False)
        self.extract = nn.Sequential(*list(self.backbone.children())[:-2],
                                     nn.BatchNorm2d(self.backbone.fc.in_features),
                                     nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.backbone.fc.in_features, 840, 3),
            nn.BatchNorm2d(840),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(840, 120, 3),
            nn.BatchNorm2d(120),
            nn.ReLU())
        self.conv3 = nn.Conv2d(120, 10, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        out0 = self.extract(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        return out3.view((-1, 10))
