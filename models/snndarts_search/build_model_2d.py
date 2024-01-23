import models.snndarts_search.cell_level_search_2d as cell_level_search
from models.snndarts_search.genotypes_2d import PRIMITIVES
from models.snndarts_search.operations_2d import *
from models.snndarts_search.decoding_formulas import Decoder

class AutoFeature(nn.Module):
    def __init__(self, frame_rate, k, args, p=0.0):
        super(AutoFeature, self).__init__()
        cell = cell_level_search.Cell
        self.cells = nn.ModuleList()
        self.p = p
        self.bits = [1, 2, 4]
        self.k = k
        self._num_layers = args.layers
        self._step = args.step
        self._block_multiplier = args.block_multiplier
        self._filter_multiplier = args.filter_multiplier
        self._initialize_alphas_betas()
        self.args = args
        f_initial = int(self._filter_multiplier)
        self._num_end = f_initial * self._block_multiplier

        self.stem0 = ConvBR(frame_rate, f_initial * self._block_multiplier, self.k, kernel_size=3, stride=1, padding=1)
        self.act_fun = ActFun_changeable().apply
        
        '''
            cell(step, block, prev_prev, prev_down, prev_same, prev_up, filter_multiplier)

            prev_prev, prev_down etc depend on tiers. If cell is in the first tier, then it won`t have prev_down.
            If cell is in the second tier, prev_down should be filter_multiplier *2, if third, then *4.(filter_multiplier is an absolute number.)
        '''

        for i in range(self._num_layers):

            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial,
                             self._filter_multiplier, self.bits, self.k, i, self.p)
                self.cells += [cell1]

            elif i == 1:
                cell2 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier,
                             self._filter_multiplier, self.bits, self.k, i, self.p)
                self.cells += [cell2]

            elif i == 2:
                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier, None,
                             self._filter_multiplier * 2, self.bits, self.k, i, self.p)
                self.cells += [cell3]

            elif i == 3:
                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier * 2,
                             self._filter_multiplier * 2, self.bits, self.k, i, self.p)
                self.cells += [cell4]

            elif i == 4:
                cell5 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             None, self._filter_multiplier * 2,
                             self._filter_multiplier * 2, self.bits, self.k, i, self.p)
                self.cells += [cell5]

            elif i == 5:
                cell6 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier * 2, None,
                             self._filter_multiplier * 4, self.bits, self.k, i, self.p)
                self.cells += [cell6]
            elif i == 6:
                cell7 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             None, self._filter_multiplier * 4,
                             self._filter_multiplier * 4, self.bits, self.k, i, self.p)
                self.cells += [cell7]
            else:
                cell8 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             None, self._filter_multiplier * 4,
                             self._filter_multiplier * 4, self.bits, self.k, i, self.p)
                self.cells += [cell8]

    def forward(self, x, param):
        layer_pruned_num = 0
        layer_model_size = 0
        layer_bit_synops = 0
        stem0 = self.stem0(x)
        # softmax on alphas and betas
        normalized_alphas = F.softmax(self.alphas, dim=-1)
        normalized_betas = F.softmax(self.betas, dim=-1)

        for layer in range(self._num_layers):
            if layer == 0:
                level3_1, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](None, None, stem0, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            elif layer == 1:
                level3_2, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](stem0, None, level3_1, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            elif layer == 2:
                level6_1, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](level3_1, level3_2, None, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            elif layer == 3:
                level6_2, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](level3_2, None, level6_1, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            elif layer == 4:
                level6_3, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](level6_1, None, level6_2, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            elif layer == 5:
                level12_1, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](level6_2, level6_3, None, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            elif layer == 6:
                level12_2, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](level6_3, None, level12_1, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops

            else:
                level12_3, cell_pruned_num, cell_model_size, cell_bit_synops = self.cells[layer](level12_1, None, level12_2, normalized_alphas, normalized_betas[layer], param)
                layer_pruned_num += cell_pruned_num
                layer_model_size += cell_model_size
                layer_bit_synops += cell_bit_synops


        return level12_3, layer_pruned_num, layer_model_size, layer_bit_synops


    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

         # initialization of alphas for architecture search
        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        # initialization of betas for mixed quantization search
        betas = (1e-3 * torch.randn(self._num_layers, len(self.bits))).clone().detach().requires_grad_(True)

        self.register_parameter('alphas', torch.nn.Parameter(alphas))
        self.register_parameter('betas', torch.nn.Parameter(betas))

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

