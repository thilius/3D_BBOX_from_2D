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
            nn.Conv2d(self.backbone.fc.in_features, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 84, 3),
            nn.BatchNorm2d(84),
            nn.ReLU())
        self.conv3 = nn.Conv2d(84, 7, 3)
        self.globalaverage = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        out0 = self.extract(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        shortcut = self.globalaverage(out0).view(-1, 2048)
        coordinate = out3.view(-1, 7)
        confidence = self.fc(shortcut)
        return torch.cat([coordinate, confidence], 1)
