import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.snndarts_search.SNN import*
OPS = {
    'skip_connect': lambda Cin, Cout, bit, k, stride, signal: (SNN_2d(Cin, Cout, bit, k, stride=stride, b=3) if signal == 1 else Identity(signal)),
    'snn_b3': lambda Cin, Cout, bit, k, stride, signal: SNN_2d(Cin, Cout, bit, k, kernel_size=3, stride=stride, b=3),
}

OPS_Retrain = {
    'skip_connect': lambda Cin, Cout, bit, k, stride, signal: (SNN_2d_Retrain(Cin, Cout, bit, k, stride=stride, b=3) if signal == 1 else Identity(signal)),
    'snn_b3': lambda Cin, Cout, bit, k, stride, signal: SNN_2d_Retrain(Cin, Cout, bit, k, kernel_size=3, stride=stride, b=3)
}


class ConvBR(nn.Module):
    def __init__(self, C_in, C_out, k, kernel_size, stride, padding, bn=True, relu=True):
        super(ConvBR, self).__init__()
        self.relu = relu
        self.use_bn = bn

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        if self.use_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)
        return x

class Identity(nn.Module):
    def __init__(self, signal):
        super(Identity, self).__init__()
        self._initialize_weights()
        self.signal = signal

    def forward(self, x):
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


