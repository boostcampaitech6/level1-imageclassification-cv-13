import torch
from torch import nn 
from torchvision.models import resnet50

class MyModel(torch.nn.Module):
    def __init__(self, num_classes=18):  # num_classes 인자 추가
        super(MyModel, self).__init__()
        self.model = resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 512)  # 마지막 레이어를 num_classes에 맞게 변경
        self.model.head = nn.Linear(512,num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.model.head(x)