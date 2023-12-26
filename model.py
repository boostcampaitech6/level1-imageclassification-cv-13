import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from facenet_pytorch import InceptionResnetV1


class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# ResNet50
class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        self.layer = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer(x)
        return self.fc(x)

# RegNet_Y_128GF
class RegNet_Y_128GFModel(nn.Module):
    def __init__(self, num_classes):
        super(RegNet_Y_128GFModel, self).__init__()
        self.layer = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer(x)
        return F.softmax(self.fc(x),dim=1)

# ViT_L_16
class ViT_L_16Model(nn.Module):
    def __init__(self, num_classes):
        super(ViT_L_16Model, self).__init__()
        self.layer = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer(x)
        return F.softmax(self.fc(x),dim=1)

class Swin_V2_BModel(nn.Module):
    def __init__(self, num_classes):
        super(Swin_V2_BModel, self).__init__()
        import timm
        self.layer = timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True) 
        for para in self.layer.parameters():
            para.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(1000, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, num_classes),
                                nn.Softmax(dim=1)
                                )

    def forward(self, x):
        x = self.layer(x)
        return self.fc(x)

class InceptionResnetV1Model(nn.Module):
    def __init__(self, num_classes):
        super(InceptionResnetV1Model, self).__init__()
        self.layer = InceptionResnetV1(pretrained='vggface2').eval()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer(x)
        return F.softmax(self.fc(x),dim=1)