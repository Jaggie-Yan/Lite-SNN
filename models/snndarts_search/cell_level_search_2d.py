from turtle import left
import torch.nn.functional as F
from models.snndarts_search.operations_2d import *
from models.snndarts_search.genotypes_2d import PRIMITIVES
import numpy as np

class MixedOp(nn.Module):
    def __init__(self, C_in, C, bit, k, stride, p, signal):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.p = p
        self.k = 4
        self.C_in = C_in
        self.C = C
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C, bit, k, stride, signal)
            if isinstance(op,Identity) and p>0:
                op = nn.Sequential(op, nn.Dropout(self.p))
            self._ops.append(op)



    def update_p(self):
        for op in self._ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p
    

    def forward(self, x, weights, bit_weights, param):
        opt_outs = []
        op_pruned_num = 0
        op_model_size = []
        op_bit_synops = []
        for i in range(len(self._ops)):
            if isinstance(self._ops[i], nn.Sequential) or isinstance(self._ops[i], Identity):
                opt_out = self._ops[i](x)
                pruned_num = 0
                model_size = 0
                bit_synops = 0
            else:
                param['snn_output'] = 'mem'
                opt_out, pruned_num, model_size, bit_synops = self._ops[i](x, bit_weights, param)
                param['snn_output'] = 'spike'
            op_pruned_num += pruned_num
            op_model_size.append(model_size)
            op_bit_synops.append(weights[i] * bit_synops)
            opt_out = weights[i] * opt_out
            opt_outs.append(opt_out)
        return sum(opt_outs), op_pruned_num, sum(op_model_size), sum(op_bit_synops)


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_fmultiplier_down, prev_fmultiplier_same,
                 filter_multiplier, bit, k, layer, p=0.0):

        super(Cell, self).__init__()

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier

        self.p = p
        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self._prev_fmultiplier_same = prev_fmultiplier_same


        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops_down = nn.ModuleList()
        self._ops_same = nn.ModuleList()

        stride_op1 = 1
        if layer == 2 or layer == 3 or layer == 5 or layer == 6:
            stride_op1 = 2

        if prev_fmultiplier_down is not None:
            c_prev_down = int(prev_fmultiplier_down * block_multiplier)
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2
                    if prev_prev_fmultiplier == -1 and j == 0:
                        op = None
                    else:
                        if j == 0:
                            op = MixedOp(self.C_prev_prev, self.C_out, bit, k, stride_op1, self.p, signal=1)
                        elif j == 1:
                            op = MixedOp(c_prev_down, self.C_out, bit, k, 2, self.p, signal=1)
                        else:
                            op = MixedOp(self.C_out, self.C_out, bit, k, 1, self.p, signal=0)
                    self._ops_down.append(op)



        if prev_fmultiplier_same is not None:
            c_prev_same = int(prev_fmultiplier_same * block_multiplier)
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 1
                    if prev_prev_fmultiplier == -1 and j == 0:
                        op = None
                    else:
                        if j == 0:
                            op = MixedOp(self.C_prev_prev,self.C_out, bit, k, stride_op1, self.p, signal=1)
                        elif j == 1:
                            op = MixedOp(c_prev_same, self.C_out, bit, k, stride, self.p, signal=1)
                        else:
                            op = MixedOp(self.C_out, self.C_out, bit, k, stride, self.p, signal=0)
                    self._ops_same.append(op)


        self._initialize_weights()
        self.mem_down = None
        self.mem_same = None
        self.act_fun = ActFun_changeable().apply

    def update_p(self):
        for op in self._ops_down:
            if op == None:
                continue
            else:
                op.p = self.p
                op.update_p()
        for op in self._ops_same:
            if op == None:
                continue
            else:
                op.p = self.p
                op.update_p()

    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear', align_corners=True)

    def forward(self, s0, s1_down, s1_same, n_alphas, n_betas, param):
        cell_pruned_num = 0
        cell_model_size = 0
        cell_bit_synops = 0
        if s1_down is not None:
            device = s1_down.device
            states = [s0, s1_down]
            offset = 0
            for i in range(self._steps):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops_down[branch_index] is None:
                        continue           
                    new_state, pruned_num, model_size, bit_synops = self._ops_down[branch_index](h, n_alphas[branch_index], n_betas, param)
                    cell_pruned_num += pruned_num
                    cell_model_size += model_size
                    cell_bit_synops += bit_synops
                    if param['is_first']:
                        self.mem_down = [torch.zeros_like(new_state,device=device)]*self._steps
                    new_states.append(new_state)
                s = sum(new_states)

                # spike out
                self.mem_down[i] = self.mem_down[i] + s
                s = self.act_fun(self.mem_down[i],3)
                self.mem_down[i] = self.mem_down[i] * decay * (1. - s) 

                offset += len(states)
                states.append(s)
            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            
            final_concates = concat_feature


        if s1_same is not None:
            device = s1_same.device
            states = [s0, s1_same]
            offset = 0
            for i in range(self._steps):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops_same[branch_index] is None:
                        continue            
                    new_state, pruned_num, model_size, bit_synops = self._ops_same[branch_index](h, n_alphas[branch_index], n_betas, param)
                    cell_pruned_num += pruned_num
                    cell_model_size += model_size
                    cell_bit_synops += bit_synops
                    if param['is_first']:
                        self.mem_same = [torch.zeros_like(new_state,device=device)]*self._steps
                    new_states.append(new_state)
                s = sum(new_states)

                # spike out
                self.mem_same[i] = self.mem_same[i] + s
                s = self.act_fun(self.mem_same[i],3)
                self.mem_same[i] = self.mem_same[i] * decay * (1. - s) 


                offset += len(states)
                states.append(s)
            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates = concat_feature
        
        return final_concates, cell_pruned_num, cell_model_size, cell_bit_synops

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
