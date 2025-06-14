import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha

        # 기존 weight는 고정
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # 학습할 저차원 행렬 A, B
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        # 스케일링 인자
        self.scaling = self.alpha / self.r

    def forward(self, x):
        # base: 기존 고정 weight 사용한 출력
        base = torch.matmul(x, self.weight.T)

        # delta: LoRA 경로 출력
        delta = torch.matmul(torch.matmul(x, self.A.T), self.B.T) * self.scaling

        return base + delta
