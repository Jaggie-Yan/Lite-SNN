import torch
import torch.nn as nn
from models.snndarts_search.build_model_2d import AutoFeature
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import math

class AutoStereo(nn.Module):
    def __init__(self, init_channels=3, args=None):
        super(AutoStereo, self).__init__()
        self.k = 0.2  # self.k is the % of weights remaining, a real number in [0,1]
        self.feature = AutoFeature(init_channels, self.k, args=args)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(192, 10)

    def update_p(self):
        self.feature.p = self.p
        self.feature.update_p()

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if 'alphas' in name]

    def CConv_parameters(self):
        return [param for name, param in self.named_parameters() if ('popup_scores' in name) or ('betas' in name)]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if
                ('alphas' not in name) and ('betas' not in name) and ('popup_scores' not in name)]

    def forward(self, input, timestamp=6):
        param = {'snn_output':'mem'}
        logits_per_time = None
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, model_pruned_num, model_size, model_bit_synops = self.feature(input, param)
            pooling_out = self.global_pooling(feature_out)
            logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1))

            if logits_per_time is None:
                logits_per_time = []
                model_bit_synops_pertime = []
            logits_per_time.append(logits_buf)
            model_bit_synops_pertime.append(model_bit_synops)

        logits_per_time = torch.stack(logits_per_time)
        model_bit_synops_pertime = torch.stack(model_bit_synops_pertime)
        logits = torch.sum(logits_per_time, dim=0) / timestamp
        model_bit_synops_total = torch.sum(model_bit_synops_pertime, dim=0)
        return logits_per_time, logits, model_pruned_num, model_size, model_bit_synops_pertime, model_bit_synops_total