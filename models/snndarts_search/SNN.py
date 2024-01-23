import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import *
from torch.nn.parameter import Parameter
import torch.autograd as autograd
import math
from torch.autograd import Variable
import copy
import numpy as np

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.2  # decay constants

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336}

class ActFun_changeable(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input)
        ctx.b = b
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            b = torch.tensor(ctx.b,device=device)
            temp = (1-torch.tanh(b*(input-0.5))**2)*b/2/(torch.tanh(b/2))
            temp[input<=0]=0
            temp[input>=1]=0
        return grad_input * temp.float(), None

# mask for pruning is obtained based on scores
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# mixed quantification weights are obtained    
class _gauss_quantize_resclaed_step(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# compressive convolution
class SharedMixQuantConv2d(nn.Module):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a trainable parameter which has the same shape as self.weight
    # gradients to self.weight, self.bias have been turned off by default
    # self.steps saves the step values of the mixed quantization operations with different bit values
    def __init__(self, inplane, outplane, bits, k, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(SharedMixQuantConv2d, self).__init__()
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.k = k
        self.popup_scores = Parameter(torch.Tensor(self.conv.weight.shape), requires_grad=True)
        score_fan_in = nn.init._calculate_correct_fan(self.conv.weight, "fan_in")
        # close to kaiming unifrom init
        self.popup_scores.data = (
            math.sqrt(6 / score_fan_in) * self.conv.weight.data / torch.max(torch.abs(self.conv.weight.data))
        )
        self.pruned_num = 0

        self.steps = []
        self.bits = bits
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input, bit_weights):
        # get the subnetwork by sorting the scores
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        self.pruned_num = (1 - adj).sum().item()

        # obtain mixed quantification weights
        mix_quant_weight = []
        conv = self.conv
        weight = conv.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            # quantified weights are weighted based on the beta value(bit_weights[i])
            scaled_quant_weight = quant_weight * bit_weights[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)

        # use only the pruned mixed quantification weights in the forward pass
        pruned_mix_quant_weight = adj * mix_quant_weight
        out = F.conv2d(
            input, pruned_mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class SNN_2d(nn.Module):

    def __init__(self, input_c, output_c, bits, k, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d, self).__init__()

        self.input_c = input_c
        self.output_c = output_c
        self.kernel_size = kernel_size
        self.bits = bits
        self.k = k
        self.conv1 = SharedMixQuantConv2d(input_c, output_c, bits, k, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)

    def forward(self, input, bit_weights, param): #20
        params = self.input_c * self.output_c * self.kernel_size * self.kernel_size
        multi_bit = self.bits[0] * bit_weights[0] + self.bits[1] * bit_weights[1] + self.bits[2] * bit_weights[2]
        model_size = self.k * multi_bit * params
        firing_rate = input.shape[0] * input.sum().item() / np.prod(input.size())
        mem_this = self.bn(self.conv1(input, bit_weights))
        bit_synops = self.k * params * multi_bit * firing_rate * mem_this.shape[2] * mem_this.shape[3]

        if param['mixed_at_mem']:
            return mem_this, self.conv1.pruned_num, model_size, bit_synops

        device = input.device
        if param['is_first']:
            self.mem = torch.zeros_like(self.conv1(input, bit_weights), device=device)

        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        return spike, self.conv1.pruned_num, model_size, bit_synops

# quantified weights are obtained
class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QuantConv2d_load(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a trainable parameter which has the same shape as self.weight
    # gradients to self.weight, self.bias have been turned off by default
    # self.step saves the step value corresponding to the searched quantization operation
    def __init__(self, inplane, outplane, bit, k, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d_load, self).__init__(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                                   dilation=dilation, groups=groups, bias=bias)
        self.k = k
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape), requires_grad=True)
        score_fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
        # Close to kaiming unifrom init
        self.popup_scores.data = (
            math.sqrt(6 / score_fan_in) * self.weight.data / torch.max(torch.abs(self.weight.data))
        )
        self.pruned_num = 0
        self.saved_num = 0
        
        self.bit = bit
        assert 0 < bit < 32
        self.step = gaussian_steps[self.bit]

    def forward(self, input):
        # get the subnetwork by sorting the scores
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        self.pruned_num = (1 - adj).sum().item()
        self.saved_num = adj.sum().item()

        # obtain quantified weights
        quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)

        # use only the pruned mixed quantified weights in the forward pass
        pruned_quant_weight = adj * quant_weight
        out = F.conv2d(
            input, pruned_quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class SNN_2d_Retrain(nn.Module):

    def __init__(self, input_c, output_c, bit, k, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d_Retrain, self).__init__()

        self.input_c = input_c
        self.output_c = output_c
        self.kernel_size = kernel_size
        self.bit = bit
        self.k = k
        self.conv1 = QuantConv2d_load(input_c, output_c, bit, k, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm2d(output_c)

    def forward(self, input, param): #20
        params = self.input_c * self.output_c * self.kernel_size * self.kernel_size
        firing_rate = input.sum().item() / np.prod(input.size())
        mem_this = self.bn(self.conv1(input))
        bit_synops = self.k * params * self.bit * firing_rate * mem_this.shape[2] * mem_this.shape[3]

        if param['mixed_at_mem']:
            return mem_this, self.conv1.pruned_num, self.conv1.saved_num * (32 - self.bit) / 8, bit_synops

        device = input.device
        if param['is_first']:
            self.mem = torch.zeros_like(self.conv1(input), device=device)

        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        return spike, self.conv1.pruned_num, self.conv1.saved_num * (32 - self.bit) / 8, bit_synops


class SNN_2d_Super(nn.Module):
    def __init__(self, input_c, output_c, bit, k, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d_Super, self).__init__()
        self.snn_optimal = SNN_2d_Retrain(input_c, output_c, bit, k, kernel_size=kernel_size, stride=stride, padding=padding, b=b)

    def forward(self, input, param):

        # print(param)
        mode = param['mode']
        if mode == 'optimal':
            spike, pruned_num, add_MB, bit_synops = self.snn_optimal(input, param)

        return spike, pruned_num, add_MB, bit_synops

class SNN_Avgpooling(nn.Module):

    def __init__(self, kernel_size,stride,padding,b=3):
        super(SNN_Avgpooling, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None

    def forward(self, input, b_mid=5): #20

        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(self.pooling(input), device=device)
        self.mem = self.mem.clone().detach() + self.pooling(input)
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        return spike

class SNN_Adaptivepooling(nn.Module):

    def __init__(self, dimension, b=3):
        super(SNN_Adaptivepooling, self).__init__()
        self.mem = None
        self.pooling = nn.AdaptiveAvgPool2d(dimension)
        self.act_fun = ActFun_changeable().apply
        self.b = b

    def forward(self, input, b_mid=5): #20

        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(self.pooling(input), device=device)
        self.mem = self.mem.clone().detach() + self.pooling(input)
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        return spike


class QuantLinear(torch.nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a trainable parameter which has the same shape as self.weight
    # gradients to self.weight, self.bias have been turned off by default
    # self.step saves the step value corresponding to the searched quantization operation
    def __init__(self, input_c, output_c, bit, k):
        super().__init__(input_c, output_c)
        self.k = k
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape), requires_grad=True)
        score_fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
        # close to kaiming unifrom init
        self.popup_scores.data = (
                math.sqrt(6 / score_fan_in) * self.weight.data / torch.max(torch.abs(self.weight.data))
        )
        self.pruned_num = 0
        self.saved_num = 0

        self.bit = bit
        assert 0 < self.bit < 32
        self.step = gaussian_steps[self.bit]

    def forward(self, x):
        # get the subnetwork by sorting the scores
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        self.pruned_num = (1 - adj).sum().item()
        self.saved_num = adj.sum().item()

        # obtain quantized conv
        quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)

        # use only the pruned quantized conv in the forward pass
        pruned_quant_weight = adj * quant_weight
        return F.linear(x, pruned_quant_weight)


class SNN_2d_fc(nn.Module):

    def __init__(self, input_c, output_c, bit, k, b=3):
        super(SNN_2d_fc, self).__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.bit = bit
        self.fc1 = QuantLinear(input_c, output_c, bit, k)
        self.mem = None
        self.bn = nn.BatchNorm1d(output_c)
        self.act_fun = ActFun_changeable().apply
        self.b = b

    def forward(self, input, b_mid=5):  # 20
        params = self.input_c * self.output_c
        output = self.bn(self.fc1(input))
        return output, self.fc1.pruned_num, self.fc1.saved_num * (32 - self.bit) / 8

    
