import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.snndarts_retrain.LEAStereo import LEAStereo
import fitlog

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--pretrained_model', type=str, default='./pretrained_model/weight.pt', help='pretrained model path')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--experiment_description', type=str, help='description of experiment')
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--filter_multiplier', type=int, default=48)
parser.add_argument('--block_multiplier', type=int, default=3)
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--net_arch', default=None, type=str)
parser.add_argument('--cell_arch', default=None, type=str)
parser.add_argument('--use_DGS', default=False, type=bool)
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

fitlog_debug = True
if fitlog_debug:
    fitlog.debug()
else:
    fitlog.commit(__file__, fit_msg=args.experiment_description)
    log_path = "logs"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    fitlog.set_log_dir(log_path)
    fitlog.create_log_folder()
    fitlog.add_hyper(args)

CIFAR_CLASSES = 10


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    leastereo = LEAStereo(init_channels=3, args=args)
    model = leastereo
    model = model.cuda()
    logging.info(f"=> loading pretrained checkpoint {args.pretrained_model}")
    ckpt = torch.load(args.pretrained_model)
    model.load_state_dict(ckpt)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    _, valid_transform = utils._data_transforms_cifar10(args)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    valid_acc = infer(valid_queue, model)
    logging.info('valid_acc %f', valid_acc)
fitlog.finish()

def infer(valid_queue, model):
    param = {'mode': 'optimal'}
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    total_pruned_num = utils.AvgrageMeter()
    total_add_MB = utils.AvgrageMeter()
    total_bit_synops = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            logits_per_time, logits, _, _, model_pruned_num, model_add_MB, model_bit_synops = model(input, param)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            total_pruned_num.update(model_pruned_num, n)
            total_add_MB.update(model_add_MB, n)
            total_bit_synops.update(model_bit_synops)

            if step % args.report_freq == 0:
                logging.info('valid %03d valid acc: %f pruned num: %f add MB: %f bit_synops: %f', step,
                             top1.avg, total_pruned_num.avg, total_add_MB.avg, total_bit_synops.avg)

    return top1.avg


if __name__ == '__main__':
    main()

