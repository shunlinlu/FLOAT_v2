import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet_OAT import ResNet34OAT
# from models.cifar100.resnet_OAT import ResNet34OATC100
# from models.tiny_imagenet.resnet_OAT import ResNet34OAT_TINY
from models.svhn.wide_resnet_OAT import WRN16_8OAT
from models.stl10.wide_resnet_OAT import WRN40_2OAT
from models.cifar10.resnet_OAT_mask import ResNet34OAT_rm

from dataloaders.cifar10 import cifar10_dataloaders
# from dataloaders.cifar100 import cifar100_dataloaders
# from dataloaders.tiny_imagenet import tiny_imagenet_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.sample_lambda import element_wise_sample_lambda, batch_wise_sample_lambda
from attacks.pgd import PGD
from admm_core import Masking, CosineDecay, LinearDecay, add_sparse_args

parser = argparse.ArgumentParser(description='cifar10 Training')
parser.add_argument('--gpu', default='1')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'stl10', 'tiny_imagenet'], help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
# OAT parameters:
#parser.add_argument('--distribution', default='disc', choices=['disc'], help='Lambda distribution')
#parser.add_argument('--lambda_choices', default=[0.0,0.1,0.2,0.3,0.4,1.0], nargs='*', type=float, help='possible lambda values to sample during training')
#parser.add_argument('--probs', default=-1, type=float, help='the probs for sample 0, if not uniform sample')
#parser.add_argument('--encoding', default='rand', choices=['none', 'onehot', 'dct', 'rand'], help='encoding scheme for Lambda')
#parser.add_argument('--dim', default=128, type=int, help='encoding dimention for Lambda')
parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
#parser.add_argument('--sampling', default='ew', choices=['ew', 'bw'], help='sampling scheme for Lambda')
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
#SK: added multiple seed support
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--decay_schedule', type=str, default='linear')
add_sparse_args(parser)
args = parser.parse_args()
args.efficient = True
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# ************************** random seed **************************
# SK: The original paper did not run with multiple seed, this is newly added.
seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# *****************************************************************
torch.backends.cudnn.benchmark = True

# data loader:
if args.dataset == 'cifar10':
    train_loader, val_loader, _ = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'cifar100':
    train_loader, val_loader, _ = cifar100_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'tiny_imagenet':
    train_loader, val_loader = tiny_imagenet_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

# model:
'''
if args.encoding in ['onehot', 'dct', 'rand']:
    FiLM_in_channels = args.dim
else: # non encoding
    FiLM_in_channels = 1
'''
if args.dataset == 'cifar10':
    model_fn = ResNet34OAT_rm
elif args.dataset == 'cifar100':
    model_fn = ResNet34OATC100
elif args.dataset == 'tiny_imagenet':
    model_fn = ResNet34OAT_TINY
elif args.dataset == 'svhn':
    model_fn = WRN16_8OAT
elif args.dataset == 'stl10':
    model_fn = WRN40_2OAT
#SK: removed , FiLM_in_channels=FiLM_in_channels from the model declaration
model = model_fn(use2BN=args.use2BN).cuda()
model = torch.nn.DataParallel(model)
# for name, p in model.named_parameters():
#     print(name, p.size())

# mkdirs:
model_str = os.path.join(model_fn.__name__)
if args.use2BN:
    model_str += '-2BN'
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-%s' % args.decay_epochs
attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%s-%d' % (args.eps, args.steps)
'''
lambda_str = '%s-%s-%s' % (args.distribution, args.sampling, args.lambda_choices)
if args.probs > 0:
    lambda_str += '-%s' % args.probs
if args.encoding in ['onehot', 'dct', 'rand']:
    lambda_str += '-%s-d%s' % (args.encoding, args.dim)
'''
#save_folder = os.path.join('./OAT_results', 'cifar10', model_str, '%s_%s_%s_%s' % (attack_str, opt_str, decay_str, lambda_str))
#SK: added seed in the folder name as well
save_folder = os.path.join('./OAT_results_v2_sparse', args.dataset, str(args.seed), str(args.density), str(args.prune), model_str, 'BN_a_for_Lm_gt0', '%s_%s_%s' % (attack_str, opt_str, decay_str))
print(save_folder)
create_dir(save_folder)
#SK: We don't use FiLM, so no need of encoding
'''
# encoding matrix:
if args.encoding == 'onehot':
    I_mat = np.eye(args.dim)
    encoding_mat = I_mat
elif args.encoding == 'dct':
    from scipy.fftpack import dct
    dct_mat = dct(np.eye(args.dim), axis=0)
    encoding_mat = dct_mat
elif args.encoding == 'rand':
    rand_mat = np.random.randn(args.dim, args.dim)
    np.save(os.path.join(save_folder, 'rand_mat.npy'), rand_mat)
    rand_otho_mat, _ = np.linalg.qr(rand_mat)
    np.save(os.path.join(save_folder, 'rand_otho_mat.npy'), rand_otho_mat)
    encoding_mat = rand_otho_mat
elif args.encoding == 'none':
    encoding_mat = None
'''
#SK: In the new code we initally use val_lambdas to only a list of 2 values, where 0 means
# evaluation on clean settings without any noise on the weights and using BN_clean,
# and 1 means evaluation on adversarial robust settings with noisy weights and using BN_adv
# Later we should support multiple val_lambdas instead of two extremes.
'''
# val_lambdas:

if args.distribution == 'disc':
    val_lambdas = args.lambda_choices
else:
    val_lambdas = [0,0.2,0.5,1]
'''
val_lambdas = [0, 0.2,0.7, 1.0]

# optimizer:
if args.opt == 'sgd':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.opt == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.decay == 'cos':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
elif args.decay == 'multisteps':
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

# load ckpt:
if args.resume:
    last_epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA \
         = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
    start_epoch = last_epoch + 1
else:
    start_epoch = 0
    # training curve lists:
    training_loss, val_TA, val_ATA, best_TA, best_ATA = [], {}, {}, {}, {}
    for val_lambda in val_lambdas:
        val_TA[val_lambda], val_ATA[val_lambda], best_TA[val_lambda], best_ATA[val_lambda] = [], [], 0, 0

# attacker:
#attacker = PGD(eps=args.eps/255, steps=args.steps, use_FiLM=True)
attacker = PGD(eps=args.eps/255, steps=args.steps)
#SK: Urgent: added a separate val_attacker with a higher attack strength to check the performance against higher attack
#while the model is trained on lower PGD strength attacks.
val_attacker = PGD(eps=args.eps/255, steps=args.steps)

if args.decay_schedule == 'cosine':
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
elif args.decay_schedule == 'linear':
    decay = LinearDecay(args.prune_rate, len(train_loader)*(args.epochs))
print('using {} decay schedule'.format(args.decay_schedule))
mask = None
mask = Masking(optimizer, decay, prune_rate=args.prune_rate, prune_mode=args.prune, \
    growth_mode=args.growth, redistribution_mode=args.redistribution, verbose=args.verbose)
mask.add_module(model, density=args.density)

## training:
for epoch in range(start_epoch, args.epochs):
    train_fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
    val_fp = open(os.path.join(save_folder, 'val_log.txt'), 'a+')
    start_time = time.time()
    ## training:
    model.train()
    requires_grad_(model, True)
    accs, accs_adv, losses, lps = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        #SK: Commenting the following sample_lambda as we don't need FiLM, however, we need num_zeros
        # to index and indicate how many input samples in a batch are to be evaluated on BN_clean with
        # no adversary added. By default for now we make num_zeros as batch_size//2.
        '''
        # sample _lambda:
        if args.sampling == 'ew':
            _lambda_flat, _lambda, num_zeros = element_wise_sample_lambda(args.distribution, args.lambda_choices, encoding_mat, 
                batch_size=args.batch_size, probs=args.probs)
        '''
        num_zeros = args.batch_size//2

        if args.use2BN:
            idx2BN = num_zeros
            #SK: assign w_noise to be False as here we first evaluate on clean images and
            # thus don;t want the weights to be noisy 
            w_noise = False
        else:
            idx2BN = None

        # logits for clean imgs:
        #SK: use model w/o noise inserted weights to evaluate the logits for clean images
        #SK: added extra argument for _lambda, and kept it 1.0
        logits = model(imgs, 1.0, w_noise, idx2BN)
        # clean loss:
        lc = F.cross_entropy(logits, labels, reduction='none')
        
        if args.efficient:
            # generate adversarial images:
            with ctx_noparamgrad_and_eval(model):
                if args.use2BN:
                    #SK: removed  _lambda=_lambda[num_zeros:], from the following function call
                    imgs_adv = attacker.attack(model, imgs[num_zeros:], labels=labels[num_zeros:], idx2BN=0)
                else:
                    #SK: removed  _lambda=_lambda[num_zeros:], from the following function call
                    imgs_adv = attacker.attack(model, imgs[num_zeros:], labels=labels[num_zeros:], idx2BN=None)
            # logits for adv imgs:
            #SK: make w_noise=True to evaluate the adversarial samples.
            w_noise = True
            logits_adv = model(imgs_adv.detach(), 1.0, w_noise, idx2BN=0)
            
            # loss and update:
            la = F.cross_entropy(logits_adv, labels[num_zeros:], reduction='none') 
            la = torch.cat([torch.zeros((num_zeros,)).cuda(), la], dim=0)
        #SK: We did not make any change here in the else section as of now, TBD
        else:
            # generate adversarial images:
            with ctx_noparamgrad_and_eval(model):
                imgs_adv = attacker.attack(model, imgs, labels=labels, _lambda=_lambda, idx2BN=idx2BN)
            # logits for adv imgs:
            logits_adv = model(imgs_adv.detach(), 1.0, w_noise, idx2BN=idx2BN)

            # loss and update:
            la = F.cross_entropy(logits_adv, labels, reduction='none')
        #SK: We give equal weight to clean image classification loss and adversarial classification loss.
        ''' 
        wc = (1-_lambda_flat)
        wa = _lambda_flat
        '''
        wc = 0.5
        wa = 0.5
        loss = torch.mean(wc * lc + wa * la) 
        optimizer.zero_grad()
        loss.backward()
        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        # get current lr:
        current_lr = scheduler.get_lr()[0]

        # metrics:
        accs.append((logits.argmax(1) == labels).float().mean().item())
        if args.efficient:
            accs_adv.append((logits_adv.argmax(1) == labels[num_zeros:]).float().mean().item())
        else:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        losses.append(loss.item())

        
        if i % 100 == 0:
            train_str = 'Epoch %d-%d | Train | Loss: %.4f, TA: %.4f, ATA: %.4f' % (
                epoch, i, losses.avg, accs.avg, accs_adv.avg)
            print(train_str)
            # print('idx2BN:', idx2BN)
            train_fp.write(train_str + '\n')
        # if i % 100 == 0:
        #     print('_lambda_flat:', _lambda_flat.size(), _lambda_flat[0:10].data.data.cpu().numpy().squeeze())
        #     print('_lambda:', _lambda.size(), _lambda[0:5,:].data.cpu().numpy().squeeze())
    # lr schedualr update at the end of each epoch:
    scheduler.step()
    if epoch < args.epochs:
        #print('Inside at end of epoch: {} and {}'.format(args.dense, args.epochs))
        mask.at_end_of_epoch()
    

    ## validation:
    model.eval()
    requires_grad_(model, False)
    print(model.training)

    eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.75*args.epochs)) # boolean
    
    if eval_this_epoch:
        val_accs, val_accs_adv = {}, {}
        for val_lambda in val_lambdas:
            val_accs[val_lambda], val_accs_adv[val_lambda] = AverageMeter(), AverageMeter()
            
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.cuda(), labels.cuda()

            for j, val_lambda in enumerate(val_lambdas):
                #SK: No need to sample any lambda
                '''
                # sample _lambda:
                if args.distribution == 'disc' and encoding_mat is not None:
                    _lambda = np.expand_dims( np.repeat(j, labels.size()[0]), axis=1 ).astype(np.uint8)
                    _lambda = encoding_mat[_lambda,:] 
                else:
                    _lambda = np.expand_dims( np.repeat(val_lambda, labels.size()[0]), axis=1 )
                _lambda = torch.from_numpy(_lambda).float().cuda()
                '''
                #priority will select which BN to choose.
                # we now as 2nd trial plan to choose BN_c for val_lambda < 0.6
                #if val_lambda < 0.6:
                #SK: made changes here to support BN_a whenever the _lambda > 0.0
                if val_lambda < 0.1:
                    priority = 0
                else:
                    priority = 1
                if args.use2BN:
                    #idx2BN = int(labels.size()[0]) if val_lambda==0 else 0
                    idx2BN = int(labels.size()[0]) if priority==0 else 0
                    if val_lambda==0:
                        w_noise = False
                    else:
                        w_noise = True
                else:
                    idx2BN = None
                # TA:
                logits = model(imgs, val_lambda, w_noise, idx2BN)
                val_accs[val_lambda].append((logits.argmax(1) == labels).float().mean().item())
                # ATA:
                # generate adversarial images:
                with ctx_noparamgrad_and_eval(model):
                    #SK: Removed _lambda=_lambda from following function call.
                    imgs_adv = val_attacker.attack(model, imgs, labels=labels, idx2BN=idx2BN)
                linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
                logits_adv = model(imgs_adv.detach(), 1.0, w_noise, idx2BN)
                val_accs_adv[val_lambda].append((logits_adv.argmax(1) == labels).float().mean().item())

    val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s' % (epoch, (time.time()-start_time), current_lr)
    if eval_this_epoch:
        val_str += ' | linf: %.4f - %.4f\n' % (torch.min(linf_norms).data, torch.max(linf_norms).data)
        for val_lambda in val_lambdas:
            val_str += 'val_lambda%s: TA: %.4f, ATA: %.4f\n' % (val_lambda, val_accs[val_lambda].avg, val_accs_adv[val_lambda].avg)
    print(val_str)
    val_fp.write(val_str + '\n')
    val_fp.close() # close file pointer

    # save loss curve:
    training_loss.append(losses.avg)
    plt.plot(training_loss)
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'training_loss.png'))
    plt.close()

    if eval_this_epoch:
        for val_lambda in val_lambdas:
            val_TA[val_lambda].append(val_accs[val_lambda].avg) 
            plt.plot(val_TA[val_lambda], 'r')
            val_ATA[val_lambda].append(val_accs_adv[val_lambda].avg)
            plt.plot(val_ATA[val_lambda], 'g')
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_acc%s.png' % val_lambda))
            plt.close()
    else:
        for val_lambda in val_lambdas:
            val_TA[val_lambda].append(val_TA[val_lambda][-1]) 
            plt.plot(val_TA[val_lambda], 'r')
            val_ATA[val_lambda].append(val_ATA[val_lambda][-1])
            plt.plot(val_ATA[val_lambda], 'g')
            plt.grid(True)
            plt.savefig(os.path.join(save_folder, 'val_acc%s.png' % val_lambda))
            plt.close()

    # save pth:
    if eval_this_epoch:
        for val_lambda in val_lambdas:
            if val_accs[val_lambda].avg >= best_TA[val_lambda]:
                best_TA[val_lambda] = val_accs[val_lambda].avg # update best TA
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_TA%s.pth' % val_lambda))
            if val_accs_adv[val_lambda].avg >= best_ATA[val_lambda]:
                best_ATA[val_lambda] = val_accs_adv[val_lambda].avg # update best ATA
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_ATA%s.pth' % val_lambda))
    save_ckpt(epoch, model, optimizer, scheduler, best_TA, best_ATA, training_loss, val_TA, val_ATA, 
        os.path.join(save_folder, 'latest.pth'))
        
