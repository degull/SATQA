import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Pooling(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        return x

class CModel(nn.Module):
    def __init__(self, out_dim=128, normalize=True):
        super().__init__()
        
        # 최신 버전의 PyTorch 사용을 가정합니다.
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        self.pre_process = nn.Sequential(*list(self.model.children())[:4])
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        self.in_dim = 2048 + 512
        self.out_dim = out_dim
        self.normalize = normalize

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1792, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.out_dim)
        )

        self.pool1 = Pooling(1024, 512)
        self.pool2 = Pooling(2048, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(7)

    def forward(self, dist, ref):
        h1 = self.pre_process(dist)
        h2 = self.pre_process(ref)

        fea1_1 = self.layer1(h1)
        fea1_2 = self.layer2(fea1_1)
        fea1_3 = self.layer3(fea1_2)
        fea1_4 = self.layer4(fea1_3)

        fea2_1 = self.layer1(h2)
        fea2_2 = self.layer2(fea2_1)
        fea2_3 = self.layer3(fea2_2)
        fea2_4 = self.layer4(fea2_3)

        fea1_1 = self.avgpool(fea1_1)
        fea1_2 = self.avgpool(fea1_2)
        fea1_3 = self.pool1(fea1_3)
        fea1_4 = self.pool2(fea1_4)

        fea2_1 = self.avgpool(fea2_1)
        fea2_2 = self.avgpool(fea2_2)
        fea2_3 = self.pool1(fea2_3)
        fea2_4 = self.pool2(fea2_4)

        fea1 = torch.cat([fea1_1, fea1_2, fea1_3, fea1_4], dim=1)
        fea2 = torch.cat([fea2_1, fea2_2, fea2_3, fea2_4], dim=1)

        if self.normalize:
            fea1 = nn.functional.normalize(fea1, dim=1)
            fea2 = nn.functional.normalize(fea2, dim=1)
        
        z1 = self.projector(fea1)
        z2 = self.projector(fea2)
        return z1, z2, fea1


if __name__ == '__main__':
    # 모델을 GPU에서 실행할지 CPU에서 실행할지 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 예시 입력 (배치 크기 5, 3채널, 224x224 크기)
    inp1 = torch.randn(5, 3, 224, 224).to(device)
    inp2 = torch.randn(5, 3, 224, 224).to(device)

    # 모델 생성 및 GPU로 이동
    model = CModel().to(device)
    
    # 모델 출력
    z1, z2, fea = model(inp1, inp2)
    
    print(z1.shape)
    print(z2.shape)
