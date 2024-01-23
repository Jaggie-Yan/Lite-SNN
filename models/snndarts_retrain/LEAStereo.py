import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.snndarts_search.SNN import *
from models.snndarts_search.decoding_formulas import network_layer_to_space
from models.snndarts_retrain.new_model_2d import newFeature
import time

class LEAStereo(nn.Module):
    def __init__(self, init_channels=3, args=None):
        super(LEAStereo, self).__init__()
        p=0.0
        self.k = 0.2  # self.k is the % of weights remaining, a real number in [0,1]
        network_path = [0,0,1,1,1,2,2,2]
        network_path = np.array(network_path)
        network_arch = network_layer_to_space(network_path)

        # Change your architecture here!
        cell_arch = [[0, 1],
                            [1, 0],
                            [4, 0],
                            [3, 0],
                            [7, 1],
                            [6, 1]]
        cell_arch = np.array(cell_arch)
        bits_arch = [2, 2, 2, 2, 1, 2, 0, 2]

        self.feature = newFeature(init_channels, network_arch, cell_arch, bits_arch, self.k, args=args)
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(576, 10, 4, self.k)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]

    def forward(self, input, param): 
        param['snn_output'] = 'mem'
        logits = None
        logits_aux_list = []
        timestamp = 5  # searched timestep
        model_bit_synops = 0
        for i in range(timestamp):
            model_pruned_num = 0
            model_add_MB = 0
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, logits_aux, pruned_num, add_MB, bit_synops = self.feature(input, param)
            model_pruned_num += pruned_num
            model_add_MB += add_MB
            model_bit_synops += bit_synops
            logits_aux_list.append(logits_aux)

 
            pooling_out = self.global_pooling(feature_out) 
            logits_buf, pruned_num, add_MB = self.classifier(pooling_out.view(pooling_out.size(0),-1)) 
            model_pruned_num += pruned_num
            model_add_MB += add_MB

            if logits is None:
                logits = []
            logits.append(logits_buf)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return test, logits, test2, logits_aux_final, model_pruned_num, model_add_MB, model_bit_synops
        else:
            return test, logits, None, None, model_pruned_num, model_add_MB, model_bit_synops


def check_spike(input):
    input = input.cpu().detach().clone().reshape(-1)
    all_01 = torch.sum(input == 0) + torch.sum(input == 1)
    print(all_01 == input.shape[0])


