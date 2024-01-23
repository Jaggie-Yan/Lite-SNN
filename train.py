import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from models.snndarts_retrain.LEAStereo import LEAStereo
import fitlog
import torch.nn.functional as F

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--resume', type=str, default=None, help='resume path')
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
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.weight_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc = 0
    start_epoch = 0
    if args.resume:
      logging.info(f"=> loading checkpoint {args.resume}")
      ckpt = torch.load(args.resume)
      start_epoch = ckpt['epoch']
      model.load_state_dict(ckpt['net'])
      optimizer.load_state_dict(ckpt['optimizer'])
      scheduler.load_state_dict(ckpt['scheduler'])
        
    for epoch in range(start_epoch, args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
        logging.info('train_acc %f', train_acc)
        fitlog.add_metric(train_acc, epoch, 'train_top1')
        fitlog.add_metric(train_obj, epoch, 'train_loss')

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        fitlog.add_metric(valid_acc, epoch, 'valid_top1')
        fitlog.add_metric(valid_obj, epoch, 'valid_loss')

        if valid_acc >= best_acc:
            best_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()
        state = {'epoch': epoch + 1,
                 'net': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()}
        utils.save_checkpoint(state, args.save)
fitlog.finish()


def train(train_queue, model, criterion, optimizer, epoch):
    param = {'mode': 'optimal'}
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    total_pruned_num = utils.AvgrageMeter()
    total_add_MB = utils.AvgrageMeter()
    total_bit_synops = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        optimizer.zero_grad()
        logits_per_time, logits, logits_aux_per_time, logits_aux, model_pruned_num, model_add_MB, model_bit_synops = model(input, param)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        total_pruned_num.update(model_pruned_num, n)
        total_add_MB.update(model_add_MB, n)
        total_bit_synops.update(model_bit_synops, n)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train %03d train loss: %e train acc: %f pruned num: %f add MB: %f bit_synops %f', step, objs.avg, top1.avg,
                         total_pruned_num.avg, total_add_MB.avg, total_bit_synops.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    param = {'mode': 'optimal'}
    objs = utils.AvgrageMeter()
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
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            total_pruned_num.update(model_pruned_num, n)
            total_add_MB.update(model_add_MB, n)
            total_bit_synops.update(model_bit_synops)

            if step % args.report_freq == 0:
                logging.info('valid %03d valid loss: %e valid acc: %f pruned num: %f add MB: %f bit_synops: %f', step, objs.avg,
                             top1.avg, total_pruned_num.avg, total_add_MB.avg, total_bit_synops.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

