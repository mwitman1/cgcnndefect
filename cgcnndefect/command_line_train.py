import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR

from .data import CIFData, CIFDataFeaturizer
from .data import collate_pool, get_train_val_test_loader
from .model import CrystalGraphConvNet
from .util import save_checkpoint, AverageMeter, class_eval, mae, Normalizer
from .model_sph_harmonics import SpookyModel, SpookyModelVectorized

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification','Fxyz'],
                    default='regression', help='complete a regression or '
                    'classification task (default: regression) '
                    '(Fxyz: do regression with forces as an additional target')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--all-elems', nargs='*', type=int, default=[0],
                    help='If training an IAP, a priori enter all possible atom'+
                         'types that can be encountered in the potential')

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')

test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument("--seed", default=0,type=int,
                    help='pytorch seed')


# New CL options added by MW
parser.add_argument('--resultdir', default='.', type=str, metavar='N',
                    help='location to write test results')
parser.add_argument('--jit', action='store_true',
                    help='Create a serialized Troch script')
parser.add_argument('--crys-spec', default=None, type=str, metavar='N',
                    help='ext of file that contains global crystal '
                         'features for each example (i.e. example.ext)'
                         'Features are concatenated to the pooled crystal'
                         'vector')
parser.add_argument('--atom-spec', default=None, type=str, metavar='N',
                    help='ext of file that contains atomic features '
                         'specific for each example (i.e. example.ext)'
                         'Features are concated to orig_atom_fea')
parser.add_argument('--csv-ext', default='', type=str,
                    help='id_prop.csv + csv-ext so that test sets can be manually '
                         'specified without recopying all of the data in a diff folder')
parser.add_argument('--model-type', default='cgcnn', type=str,
                    choices=['cgcnn','spooky'])
parser.add_argument('--njmax', default=75, type=int, 
                    help='Max num nbrs for sph harm featurization')
parser.add_argument('--init-embed-file',default='atom_init.json', type=str,
                    help='file for the initial atom embeddings (based only on elemental identity)')


args = parser.parse_args(sys.argv[1:])

with open(os.path.join(args.resultdir,'parameters_CGCNNtrain.json'),'w') as fp:
    json.dump(vars(args),fp)

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error
    torch.manual_seed(args.seed)

    # load data
    print(args.task)
    #torch.multiprocessing.set_sharing_strategy('file_system')
    print(args.task=='Fxyz')
    dataset = CIFData(*args.data_options,
                      args.task=='Fxyz',            # MW: to remove
                      args.all_elems,               # MW: needed for computing ZBL
                      crys_spec = args.crys_spec,   # MW: if global crystal features available
                      atom_spec = args.atom_spec,   # MW: if local/atom features available
                      csv_ext = args.csv_ext,       # MW: if using a specific id_prop.csv.*
                      model_type = args.model_type, # MW: if using non-CGCNN model
                      njmax = args.njmax,           # MW: max nbrs for sph_harm
                      init_embed_file = args.init_embed_file) # MW: choosing specific file for initial atom embed
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
        normalizer_Fxyz = None
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        #_, sample_target, _ = collate_pool(sample_data_list) #<- Fxyz mod
        _, sample_target, sample_target_Fxyz, _ =\
            collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)
        if args.task == 'Fxyz':
            raise NotImplemented("Forces not implemented yet")
            normalizer_Fxyz = Normalizer(sample_target_Fxyz)
        else:
            normalizer_Fxyz = Normalizer(sample_target) # dummy object for compatibility

    # build model
    #structures, _, _, = dataset[0] #<- Fxyz mod
    structures, _, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    if args.crys_spec is not None:
        global_fea_len = len(structures[7])
    else:
        global_fea_len = 0

    if args.model_type == 'cgcnn':
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True if args.task ==
                                                           'classification' else False,
                                    Fxyz=True if args.task == 'Fxyz' else False, #MW
                                    all_elems=args.all_elems, #MW
                                    global_fea_len=global_fea_len) #MW
    elif args.model_type == 'spooky':
        if args.njmax > 0:
            model = SpookyModelVectorized(orig_atom_fea_len,
                                          atom_fea_len = args.atom_fea_len,
                                          n_conv = args.n_conv,
                                          h_fea_len = args.h_fea_len,
                                          n_h = args.n_h,
                                          global_fea_len = global_fea_len,
                                          njmax = args.njmax) #MW
        else:
            # TODO: to be discontinued once final testing done
            model = SpookyModel(orig_atom_fea_len,
                                atom_fea_len = args.atom_fea_len,
                                n_conv = args.n_conv,
                                h_fea_len = args.h_fea_len,
                                n_h = args.n_h,
                                global_fea_len = global_fea_len) #MW
        
                            


    print("Number trainable params: %d"%\
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
    #                                              base_lr=args.lr, 
    #                                              max_lr=args.lr*5,
    #                                              step_size_up=200,
    #                                              cycle_momentum=False)

    # Pickle the CIFData object so the exact same settings
    # can be used in predict mode
    with open(os.path.join(args.resultdir,'dataset.pth.tar'),'wb') as f:
        pickle.dump(dataset, f) 

    f = open(os.path.join(args.resultdir,"train.log"),"w")

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        summary = train(train_loader, model, criterion, 
              optimizer, epoch, normalizer, normalizer_Fxyz)
        f.write(summary)

        # evaluate on validation set
        mae_error,summary = validate(val_loader, model, criterion, normalizer, 
                             normalizer_Fxyz)
        f.write(summary)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()


        # remember the best mae_eror and save checkpoint
        if args.task == 'regression' or args.task=='Fxyz':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'normalizer_Fxyz': normalizer_Fxyz.state_dict(),
            'args': vars(args)
        }, is_best, args.resultdir)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load(os.path.join(args.resultdir,'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['state_dict'])
    mae,summary = validate(test_loader, model, criterion, normalizer, normalizer_Fxyz, test=True)

    if args.jit:
        sm = torch.jit.script(model)
        sm.save(os.path.join(args.resultdir,"model_best.pt"))

        sm1 = torch.jit.load(os.path.join(args.resultdir,'model_best.pt'))
        print(sm1.dataset1.foo())

        #sm1 = torch.jit.script(dataset1)
        #sm1.save ('dataset1.pt')

    f.write('---------Evaluate Model on Test Set---------------\n')
    f.write(summary)
    f.close()

def train(train_loader, model, criterion, optimizer, epoch, 
          normalizer, normalizer_Fxyz):
    summary=""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = None
    elif args.task =='Fxyz':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_Fxyz, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            # TODO need to adapt for CUDA
            raise NotImplemented("Cuda operation not implemented yet")
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            if args.model_type == 'cgcnn':
                input_var = (Variable(input[0]), # batch_atom_fea
                             Variable(input[1]), # batch_nbr_fea
                             input[2],           # batch_nbr_fea_idx
                             input[3],           # crystal_atom_idx
                             input[4],           # MW: batch_atom_type 
                             input[5],           # MW: batch_nbr_type
                             input[6],           # MW: batch_nbr_dist
                             input[7],           # MW: batch_pair_type
                             input[8])           # MW: batch_global_fea
            elif args.model_type == 'spooky':
                input_var = (Variable(input[0]), # batch_atom_fea
                             input[9],           # batch_nbr_fea_idx_all
                             input[3],           # crystal_atom_idx
                             input[10],          # batch_gs_fea
                             input[11],          # batch_gp_fea
                             input[12],          # batch_gd_fea
                             input[8])           # batch_global_fea

            if args.all_elems != [0]:
                crys_rep_ener = model.compute_repulsive_ener(input[3],
                                                             input[4],
                                                             input[5],
                                                             input[6])
            else:
                crys_rep_ener = torch.zeros(target.shape)
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target-crys_rep_ener)
            target_Fxyz_normed = None 
        elif args.task =='Fxyz':
            target_normed = normalizer.norm(target)
            #print(target.shape)
            #print(target_normed.shape)
            # need to flatten all of the atom environments in each struct
            # s.t. first dim is total atom environments in batch
            target_Fxyz=torch.flatten(target_Fxyz,start_dim=0,end_dim=1)
            #print(target_Fxyz.shape)
            target_Fxyz_normed = normalizer_Fxyz.norm(target_Fxyz)
            #print(target_Fxyz_normed.shape)
        else:
            target_normed = target.view(-1).long()
            target_Fxyz_normed = None 

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
            # Fxyz TODO for cuda
        else:
            target_var = Variable(target_normed)
            if args.task=='Fxyz':
                target_var_Fxyz = Variable(target_Fxyz_normed)

        # compute output = target, target_Fxyz (if Fxyz specified else None)
        output = model(*input_var)
        
        #loss = criterion(output, target_var) #<- Fxyz mod
        loss_orig = criterion(output[0], target_var)
        alpha=1
        if args.task == 'Fxyz':
            #print(output[1].shape)
            #print(target_var_Fxyz.shape)
            loss_Fxyz = criterion(output[1],target_var_Fxyz)
            #print(loss_Fxyz.shape)
            loss = loss_orig+alpha*loss_Fxyz
            #print(loss_orig,loss_Fxyz)
        else:
            loss = loss_orig

        # measure accuracy and record loss
        if args.task == 'regression':
            #mae_error = mae(normalizer.denorm(output.data.cpu()), target) #<-Fxyz change
            mae_error = mae(normalizer.denorm(output[0].data.cpu())+crys_rep_ener, target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        elif args.task == 'Fxyz':
            mae_error = mae(normalizer.denorm(output[0].data.cpu()), target)
            mae_Fxyz_error = mae(normalizer_Fxyz.denorm(output[1].data.cpu()),
                                 target_Fxyz)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            mae_Fxyz_errors.update(mae_Fxyz_error, target_Fxyz.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        if name == 'r12coeffs':
        #            print(name, param.data, param.grad)

        optimizer.step()



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                summary = 'Epoch: [{0}][{1}/{2}]\t'\
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                #)
            elif args.task == 'Fxyz':
                summary='Epoch: [{0}][{1}/{2}]\t'\
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'\
                      'MAE_Fxyz {mae_Fxyz_errors.val:.3f} '\
                      '({mae_Fxyz_errors.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors,
                    mae_Fxyz_errors=mae_Fxyz_errors)
                #)
            else:
                summary='Epoch: [{0}][{1}/{2}]\t'\
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'\
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'\
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'\
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'\
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                #)
            print(summary)
    return summary+'\n'
    


def validate(val_loader, model, criterion, normalizer, normalizer_Fxyz, test=False):
    summary=""
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = None
    elif args.task =='Fxyz':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        test_targets_Fxyz = []
        test_preds_Fxyz = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, target_Fxyz, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                if args.model_type == 'cgcnn':
                    input_var = (Variable(input[0]), # batch_atom_fea
                                 Variable(input[1]), # batch_nbr_fea
                                 input[2],           # batch_nbr_fea_idx
                                 input[3],           # crystal_atom_idx
                                 input[4],           # MW: batch_atom_type 
                                 input[5],           # MW: batch_nbr_type
                                 input[6],           # MW: batch_nbr_dist
                                 input[7],           # MW: batch_pair_type
                                 input[8])           # MW: batch_global_fea
                elif args.model_type == 'spooky':
                    input_var = (Variable(input[0]), # batch_atom_fea
                                 input[9],           # batch_nbr_fea_idx_all
                                 input[3],           # crystal_atom_idx
                                 input[10],          # batch_gs_fea
                                 input[11],          # batch_gp_fea
                                 input[12],          # batch_gd_fea
                                 input[8])           # batch_global_fea

            if args.all_elems != [0]:
                crys_rep_ener = model.compute_repulsive_ener(input[3],
                                                             input[4],
                                                             input[5],
                                                             input[6])
            else:
                crys_rep_ener = torch.zeros(target.shape)

        if args.task == 'regression':
            target_normed = normalizer.norm(target-crys_rep_ener)
            target_Fxyz_normed = None 
        elif args.task =='Fxyz':
            target_normed = normalizer.norm(target)
            target_Fxyz=torch.flatten(target_Fxyz,start_dim=0,end_dim=1)
            target_Fxyz_normed = normalizer_Fxyz.norm(target_Fxyz)
        else:
            target_normed = target.view(-1).long()
            target_Fxyz_normed = None 

        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)
                if args.task=='Fxyz':
                    target_var_Fxyz = Variable(target_Fxyz_normed)

        # compute output
        output = model(*input_var)
        #loss = criterion(output, target_var) $<- Fxyz mod
        loss_orig = criterion(output[0], target_var)
        alpha=1
        if args.task == 'Fxyz':
            loss_Fxyz = criterion(output[1],target_var_Fxyz)
            loss = loss_orig+alpha*loss_Fxyz
        else:
            loss = loss_orig


        # measure accuracy and record loss
        if args.task == 'regression':
            #mae_error = mae(normalizer.denorm(output.data.cpu()), target) #<- Fxyz mod
            mae_error = mae(normalizer.denorm(output[0].data.cpu())+crys_rep_ener, target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                #test_pred = normalizer.denorm(output.data.cpu()) #<-Fxyz mod
                test_pred = normalizer.denorm(output[0].data.cpu())+crys_rep_ener
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        elif args.task == 'Fxyz':
            mae_error = mae(normalizer.denorm(output[0].data.cpu()), target)
            mae_Fxyz_error = mae(normalizer_Fxyz.denorm(output[1].data.cpu()),
                                 target_Fxyz)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            mae_Fxyz_errors.update(mae_Fxyz_error, target_Fxyz.size(0))
            if test:
                test_pred = normalizer.denorm(output[0].data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

                test_pred_Fxyz = normalizer_Fxyz.denorm(output[1].data.cpu())
                test_target_Fxyz = target_Fxyz
                test_preds_Fxyz.append(test_pred_Fxyz.view(-1).tolist())
                test_targets_Fxyz.append(test_target_Fxyz.view(-1).tolist())
                #print(test_pred_Fxyz.shape)
                #print(test_target_Fxyz.shape)
                #print(test_preds_Fxyz)
                #print(test_targets_Fxyz)
        
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                summary1='Test: [{0}/{1}]\t'\
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors)
            elif args.task == 'Fxyz':
                summary1='Test: [{0}/{1}]\t'\
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'\
                      'MAE_Fxyz {mae_Fxyz_errors.val:.3f} '\
                      '({mae_Fxyz_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors,mae_Fxyz_errors=mae_Fxyz_errors)
            else:
                summary1='Test: [{0}/{1}]\t'\
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'\
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'\
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'\
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'\
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores)
            print(summary1)
            summary+=summary1+'\n'
            

    if test:
        star_label = '**'
        import csv
        with open(os.path.join(args.resultdir,'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        if args.task == 'Fxyz':
            with open(os.path.join(args.resultdir,'test_results_Fxyz.csv'), 'w') as f:
                for cif_id, target_Fxyz, pred_Fxyz in zip(test_cif_ids, 
                                                          test_targets_Fxyz,
                                                          test_preds_Fxyz):
                    for i,target_F, pred_F in zip(range(len(target_Fxyz)),
                                                  target_Fxyz,
                                                  pred_Fxyz):
                        f.write("%s,%d,%.5f,%.5f\n"%(cif_id, 
                                                   int(np.floor(i/3)),
                                                   target_F,
                                                   pred_F))

    else:
        star_label = '*'
    if args.task == 'regression':
        summary1=' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors)
        print(summary1)
        return mae_errors.avg,summary+summary1+'\n'
    elif args.task == 'Fxyz':
        summary1=' {star} MAE {mae_errors.avg:.3f} '\
              'MAE_Fxyz {mae_Fxyz_errors.avg:.3f}'.format(
                    star=star_label,
                    mae_errors=mae_errors,
                    mae_Fxyz_errors=mae_Fxyz_errors)
        print(summary1)
        return mae_errors.avg+mae_Fxyz_errors.avg,summary+summary1+'\n'
    else:
        summary1=' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores)
        print(summary1)
        return auc_scores.avg, summary+summary1+'\n'


#class Normalizer(object):
#    """Normalize a Tensor and restore it later. """
#
#    def __init__(self, tensor):
#        """tensor is taken as a sample to calculate the mean and std"""
#        self.mean = torch.mean(tensor)
#        self.std = torch.std(tensor)
#
#    def norm(self, tensor):
#        return (tensor - self.mean) / self.std
#
#    def denorm(self, normed_tensor):
#        return normed_tensor * self.std + self.mean
#
#    def state_dict(self):
#        return {'mean': self.mean,
#                'std': self.std}
#
#    def load_state_dict(self, state_dict):
#        self.mean = state_dict['mean']
#        self.std = state_dict['std']
#
#
#def mae(prediction, target):
#    """
#    Computes the mean absolute error between prediction and target
#
#    Parameters
#    ----------
#
#    prediction: torch.Tensor (N, 1)
#    target: torch.Tensor (N, 1)
#    """
#    return torch.mean(torch.abs(target - prediction))
#
#
#def class_eval(prediction, target):
#    prediction = np.exp(prediction.numpy())
#    target = target.numpy()
#    pred_label = np.argmax(prediction, axis=1)
#    target_label = np.squeeze(target)
#    if not target_label.shape:
#        target_label = np.asarray([target_label])
#    if prediction.shape[1] == 2:
#        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
#            target_label, pred_label, average='binary')
#        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
#        accuracy = metrics.accuracy_score(target_label, pred_label)
#    else:
#        raise NotImplementedError
#    return accuracy, precision, recall, fscore, auc_score
#
#
#class AverageMeter(object):
#    """Computes and stores the average and current value"""
#
#    def __init__(self):
#        self.reset()
#
#    def reset(self):
#        self.val = 0
#        self.avg = 0
#        self.sum = 0
#        self.count = 0
#
#    def update(self, val, n=1):
#        self.val = val
#        self.sum += val * n
#        self.count += n
#        self.avg = self.sum / self.count
#
#
#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#    torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, 'model_best.pth.tar')
#
#
#def adjust_learning_rate(optimizer, epoch, k):
#    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
#    assert type(k) is int
#    lr = args.lr * (0.1 ** (epoch // k))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr


if __name__ == '__main__':
    main()
