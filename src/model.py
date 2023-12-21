import torch
import torch.nn as nn
import torchvision.models as models
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        
        ## vit_base_patch16_384
        self.model = timm.create_model('vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    model = BaseModel(18)
    print(model)
