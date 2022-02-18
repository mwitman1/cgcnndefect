import argparse
import os
import shutil
import sys
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .data import CIFData
from .data import collate_pool
from .model import CrystalGraphConvNet
from .model_sph_harmonics import SpookyModel, SpookyModelVectorized
from .util import save_checkpoint, AverageMeter, class_eval, mae, Normalizer

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-CIFdatapath',type=str,default=None,
                    help='resuse the CIFdata object initialized during trainig')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resultdir', default='.', type=str, metavar='N',
                        help='location to write test results')
#parser.add_argument('--crys-spec', default=None, type=str, metavar='N',
#                    help='ext of file that contains global crystal '
#                         'features for each example (i.e. example.ext)'
#                         'Features are concatenated to the pooled crystal'
#                         'vector')
#parser.add_argument('--atom-spec', default=None, type=str, metavar='N',
#                    help='ext of file that contains atomic features '
#                         'specific for each example (i.e. example.ext)'
#                         'Features are concated to orig_atom_fea')
parser.add_argument('--csv-ext', default='', type=str,
                    help='id_prop.csv + csv-ext so that test sets can be manually '
                         'specified without recopying all of the data in a diff folder')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args, best_mae_error

    # load data
    if args.CIFdatapath is not None:
        with open(args.CIFdatapath,'rb') as f:
            dataset = pickle.load(f)
            dataset.csv_ext = args.csv_ext
            dataset.reset_root(args.cifpath)
    else:
        dataset = CIFData(args.cifpath)
    #dataset.csv_ext = args.csv_ext
    #dataset.reload_data()
    

    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)
    # build model
    #structures, _, _ = dataset[0]
    structures, _, _, _ = dataset[0] #<- Fxyz mod
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    if model_args.crys_spec is not None:
        global_fea_len = len(structures[7])
    else:
        global_fea_len = 0
    print("Potential applicable to: ", dataset.all_elems)

    if model_args.model_type == 'cgcnn':
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                    'classification' else False,
                                    Fxyz=True if model_args.task == 'Fxyz' else False, #MW
                                    all_elems=model_args.all_elems, #MW
                                    global_fea_len=global_fea_len) #MW
    elif model_args.model_type == 'spooky':
        if model_args.njmax > 0:
            model = SpookyModelVectorized(orig_atom_fea_len,
                                          atom_fea_len = model_args.atom_fea_len,
                                          n_conv = model_args.n_conv,
                                          h_fea_len = model_args.h_fea_len,
                                          n_h = model_args.n_h,
                                          global_fea_len = global_fea_len,
                                          njmax = model_args.njmax) #MW 
        else:
            # TODO: to be discontinued once final testing done
            raise ValueError('Only use vectorized model, specified by njmax > 0!')
            model = SpookyModel(orig_atom_fea_len,
                                atom_fea_len = args.atom_fea_len,
                                n_conv = args.n_conv,
                                h_fea_len = args.h_fea_len,
                                n_h = args.n_h,
                                global_fea_len = global_fea_len) #MW 

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    # if args.optim == 'SGD':
    #     optimizer = optim.SGD(model.parameters(), args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    # elif args.optim == 'Adam':
    #     optimizer = optim.Adam(model.parameters(), args.lr,
    #                            weight_decay=args.weight_decay)
    # else:
    #     raise NameError('Only SGD or Adam is allowed as --optim')

    normalizer = Normalizer(torch.zeros(3))
    # TODO check if this is correct
    normalizer_Fxyz = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    validate(test_loader, model, criterion, normalizer, normalizer_Fxyz, test=True)


def validate(val_loader, model, criterion, normalizer, normalizer_Fxyz, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = None
    elif model_args.task == 'Fxyz':
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
        with torch.no_grad():
            if args.cuda:
                # TODO update for CUDA
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                if model_args.model_type == 'cgcnn':
                    input_var = (Variable(input[0]), # batch_atom_fea
                                 Variable(input[1]), # batch_nbr_fea
                                 input[2],           # batch_nbr_fea_idx
                                 input[3],           # crystal_atom_idx
                                 input[4],           # MW: batch_atom_type 
                                 input[5],           # MW: batch_nbr_type
                                 input[6],           # MW: batch_nbr_dist
                                 input[7],           # MW: batch_pair_type
                                 input[8])           # MW: batch_global_fea
                elif model_args.model_type == 'spooky':
                    input_var = (Variable(input[0]), # batch_atom_fea
                                 input[9],           # batch_nbr_fea_idx_all
                                 input[3],           # crystal_atom_idx
                                 input[10],          # batch_gs_fea
                                 input[11],          # batch_gp_fea
                                 input[12],          # batch_gd_fea
                             input[8])           # batch_global_fea

                if model_args.all_elems != [0]:
                    crys_rep_ener = model.compute_repulsive_ener(input[3],
                                                                 input[4],
                                                                 input[5],
                                                                 input[6])
                else:
                    crys_rep_ener = torch.zeros(target.shape)

        if model_args.task == 'regression':
            target_normed = normalizer.norm(target-crys_rep_ener)
            target_Fxyz_normed = None
        elif model_args.task == 'Fxyz':
            target_normed = normalizer.norm(target)
            target_Fxyz=torch.flatten(target_Fxyz,start_dim=0,end_dim=1)
            target_Fxyz_normed = normalizer_Fxyz.norm(target_Fxyz)
        else:
            target_normed = target.view(-1).long()
            target_Fxyz_normed = None
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
                # Fxyz TODO for cuda
            else:
                target_var = Variable(target_normed)
                if model_args.task=='Fxyz':
                    target_var_Fxyz = Variable(target_Fxyz_normed)

        # compute output
        output = model(*input_var)
        #loss = criterion(output, target_var) #<-Fxyz mod
        loss_orig = criterion(output[0], target_var)
        alpha=1
        if model_args.task == 'Fxyz':
            loss_Fxyz = criterion(output[1], target_var_Fxyz)
            loss = loss_orig+alpha*loss_Fxyz
        else:
            loss = loss_orig
        #print(output[0].data.cpu(),
        #      normalizer.denorm(output[0].data.cpu()),
        #      crys_rep_ener,
        #      normalizer.denorm(output[0].data.cpu())+crys_rep_ener)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            #mae_error = mae(normalizer.denorm(output.data.cpu()), target) #<-Fxyz change
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
        elif model_args.task == 'Fxyz':
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

        else:
            accuracy, precision, recall, fscore, auc_score =\
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
            if model_args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors))
            elif model_args.task == 'Fxyz':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                      'MAE_Fxyz {mae_Fxyz_errors.val:.3f} '
                      '({mae_Fxyz_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors,mae_Fxyz_errors=mae_Fxyz_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       accu=accuracies, prec=precisions, recall=recalls,
                       f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open(os.path.join(args.resultdir,'all_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        if model_args.task == 'Fxyz':                                                 
            with open(os.path.join(args.resultdir,'all_results_Fxyz.csv'), 'w') as f:
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
    if model_args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    elif model_args.task == 'Fxyz':
        print(' {star} MAE {mae_errors.avg:.3f} '
              'MAE_Fxyz {mae_Fxyz_errors.avg:.3f}'.format(
                    star=star_label,
                    mae_errors=mae_errors,
                    mae_Fxyz_errors=mae_Fxyz_errors))
        return mae_errors.avg+mae_Fxyz_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


#class Normalizer(object):
#    """Normalize a Tensor and restore it later. """
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


if __name__ == '__main__':
    main()
