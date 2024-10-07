import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent_Sup(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature
    
    def forward(self, z_i, z_j, label):
        batch_size = z_i.shape[0]
        N = 2 * batch_size  # 배치 크기의 두 배로 설정
        epsilon = 1e-8  # 작은 값 추가

        # z_i와 z_j 결합
        z = torch.cat([z_i, z_j], dim=0)

        # label도 두 배로 확장 (양성-음성 쌍에 대응)
        label = torch.cat([label, label], dim=0)

        # 유사성 계산
        z = F.normalize(z, p=2, dim=-1)
        sim = torch.mm(z, z.T) / self.t  # 유사성 행렬 계산

        # 양성 및 음성 마스크 생성
        mask_pos = torch.eye(N, dtype=torch.float32, device=sim.device)  # N x N 양성 마스크
        mask_pos.fill_diagonal_(0)  # 대각선은 0으로 설정
        
        mask_neg = torch.ones_like(mask_pos, device=sim.device)  # 음성 마스크 생성
        mask_neg.fill_diagonal_(0)  # 음성 마스크에서 대각선 0 설정

        # 손실 계산
        numerator = torch.sum(torch.exp(sim) * mask_pos, dim=-1)
        denominator = torch.sum(torch.exp(sim) * mask_neg, dim=-1) + epsilon  # epsilon 추가

        # 손실 함수 적용
        loss = -torch.log((numerator + epsilon) / denominator)  # 분자에도 epsilon 추가
        loss = torch.mean(loss)

        return loss



#        # remove the sim column which not contain same class
#        idx = torch.where(pos_cnt > 0)
#        numerator = torch.sum(torch.exp(sim) * mask_pos, dim=-1)
#        numerator = numerator[idx] / pos_cnt[idx]
#        denominator = denominator[idx]
#
#        loss = torch.mean(torch.log(denominator) - torch.log(numerator))
#
#        return loss


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    inp1 = torch.randn(3, 12)
    inp2 = torch.randn(3, 12)

    label = torch.tensor([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

    loss_fn = NT_Xent_Sup(temperature=0.1)
    print(loss_fn(inp1, inp2, label))
