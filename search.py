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
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.autograd import Variable
from models.snndarts_search.build_model import AutoStereo
import fitlog

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--arch_lr', default=0.001, type=float)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=3, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--filter_multiplier', type=int, default=12)
parser.add_argument('--block_multiplier', type=int, default=4)
parser.add_argument('--step', type=int, default=4)
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--CIFAR10', action='store_true', default=False, help='CIFAR 10')
parser.add_argument('--CIFAR100', action='store_true', default=False, help='CIFAR 100')
parser.add_argument('--DVSCIFAR10', action='store_true', default=False, help='DVS CIFAR 100')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--experiment_description', type=str, default="SNN_DARTS", help='description of experiment')
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


# fitlog_debug = False
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

if args.CIFAR10 or args.DVSCIFAR10:
    CIFAR_CLASSES = 10
elif args.CIFAR100:
    CIFAR_CLASSES = 100
else:
    raise NotImplementedError


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    # initialize process group
    dist.init_process_group(backend='nccl')
    dist.barrier()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = AutoStereo(args.init_channels, args=args)
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # optimizer for weights updates
    optimizer = torch.optim.SGD(
        model.module.weight_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # optimizer for scores and betas updates
    CConv_optimizer = torch.optim.Adam(model.module.CConv_parameters(),
                                       lr=1e-2, betas=(0.9, 0.999),
                                       weight_decay=args.arch_weight_decay)
    # optimizer for alphas updates
    architect_optimizer = torch.optim.Adam(model.module.arch_parameters(),
                                           lr=args.arch_lr, betas=(0.9, 0.999),
                                           weight_decay=args.arch_weight_decay)
    # optimizer for psi updates
    psi = (1e-3 * torch.randn(6)).clone().detach().requires_grad_(True).to(device)
    model.module.register_parameter('psi', Parameter(psi))
    loss_parameters = [param for name, param in model.module.named_parameters() if 'psi' in name]
    psi_optimizer = torch.optim.SGD(
        loss_parameters,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)


    if args.CIFAR10:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.CIFAR100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.DVSCIFAR10:
        train_data, valid_data = utils.build_dvscifar(path=args.data)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  # 25000
    train_data1 = torch.utils.data.Subset(train_data, indices[0:split])
    train_data2 = torch.utils.data.Subset(train_data, indices[split:num_train])
    train_sampler = DistributedSampler(train_data1, num_replicas=dist.get_world_size(), rank=args.local_rank)
    valid_sampler = DistributedSampler(train_data2, num_replicas=dist.get_world_size(), rank=args.local_rank)

    if args.CIFAR10 or args.CIFAR100:
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True, num_workers=4)  # 25000

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=valid_sampler,
            pin_memory=True, num_workers=4)  # 25000

    elif args.DVSCIFAR10:
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=2)  # 9000

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=20,
            pin_memory=True, num_workers=2)  # 1000

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    psi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        psi_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    start_epoch = 0
    if args.resume:
        logging.info(f"=> loading checkpoint {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        CConv_optimizer.load_state_dict(ckpt['CConv_optimizer'])
        architect_optimizer.load_state_dict(ckpt['architect_optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model,
                                    criterion, optimizer,
                                    CConv_optimizer,
                                    architect_optimizer, psi_optimizer, lr, epoch)

        logging.info('train_acc %f', train_acc)
        fitlog.add_metric(train_acc, epoch, 'train_top1')
        fitlog.add_metric(train_obj, epoch, 'train_loss')

        utils.save(model, os.path.join(args.save, 'epoch_%s.pt' % epoch))
        state = {'epoch': epoch + 1,
                 'net': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'CConv_optimizer': CConv_optimizer.state_dict(),
                 'architect_optimizer': architect_optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()}
        utils.save_checkpoint(state, args.save)
        scheduler.step()
        if epoch >= 5:
          psi_scheduler.step()
fitlog.finish()


def train(train_queue, valid_queue, model, criterion, optimizer, CConv_optimizer, architect_optimizer, psi_optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    total_pruned_num = utils.AvgrageMeter()
    total_model_size = utils.AvgrageMeter()
    total_bit_synops = utils.AvgrageMeter()

    model.train()
    valid_iter = iter(valid_queue)

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = Variable(input, requires_grad=False).to(torch.cuda.current_device())
        target = Variable(target, requires_grad=False).to(torch.cuda.current_device())

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(valid_iter)
        input_search = Variable(input_search, requires_grad=False).to(torch.cuda.current_device())
        target_search = Variable(target_search, requires_grad=False).to(torch.cuda.current_device())

        optimizer.zero_grad()
        CConv_optimizer.zero_grad()
        architect_optimizer.zero_grad()
        if epoch >= 5:
          psi_optimizer.zero_grad()

        logits_per_time, logits, model_pruned_num, model_size, model_bit_synops_pertime, model_bit_synops_total = model(input)
        loss = utils.Our_loss(logits_per_time, target, criterion, model_size, model_bit_synops_pertime, model.module.psi, epoch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        CConv_optimizer.zero_grad()
        architect_optimizer.zero_grad()
        if epoch >= 5:
          psi_optimizer.zero_grad()

        ##### search forward #####
        logits_per_time, logits, model_pruned_num, model_size, model_bit_synops_pertime, model_bit_synops_total = model(input)
        # loss = criterion(logits, target) + model_cost * 1e-10
        loss = utils.Our_loss(logits_per_time, target, criterion, model_size, model_bit_synops_pertime, model.module.psi, epoch)
        loss.backward()
        CConv_optimizer.step()
        architect_optimizer.step()
        # warmup epochs
        if epoch >= 5:
          psi_optimizer.step()
          psi_optimizer.zero_grad()
        optimizer.zero_grad()
        CConv_optimizer.zero_grad()
        architect_optimizer.zero_grad()
        ##### end search forward #####

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        total_pruned_num.update(model_pruned_num, n)
        total_model_size.update(model_size, n)
        total_bit_synops.update(model_bit_synops_total, n)

        if step % args.report_freq == 0:
            logging.info('train %03d train loss: %e train acc: %f pruned num: %f model size: %f model bit synops: %f', step, objs.avg, top1.avg, total_pruned_num.avg, total_model_size.avg, total_bit_synops.avg)
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

