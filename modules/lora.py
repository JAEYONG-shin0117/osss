import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha

        # 기존 weight는 동결
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # bias 추가 (기존 nn.Linear와 호환)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA 학습 파라미터
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.scaling = self.alpha / self.r

    def forward(self, x):
        base = x @ self.weight.T
        delta = (x @ self.A.T) @ self.B.T * self.scaling
        output = base + delta
        if self.bias is not None:
            output += self.bias
        return output
