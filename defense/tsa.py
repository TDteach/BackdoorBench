'''
Anti-backdoor learning: Training clean models on poisoned data.
This file is modified based on the following source:
link : https://github.com/bboylyg/ABL.
The defense method is called abl.
@article{li2021anti,
            title={Anti-backdoor learning: Training clean models on poisoned data},
            author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={14900--14912},
            year={2021}
            }
The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. abl defense:
        a. pre-train model
        b. isolate the special data(loss is low) as backdoor data
        c. unlearn the backdoor data and learn the remaining data
    4. test the result and get ASR, ACC, RC 
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import logging
from defense.base import defense

from torch.utils.data import DataLoader

from utils.trainer_cls import Metric_Aggregator, all_acc, test_given_dataloader_on_mix

from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import dataset_wrapper_with_transform

from utils.useful_tools import register_hook_before_final
from utils.useful_tools import RandomDataset


class LGALoss(nn.Module):
    def __init__(self, gamma, criterion):
        super(LGALoss, self).__init__()
        self.gamma = gamma
        self.criterion = criterion
        return
    
    def forward(self,output,target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = torch.sign(loss - self.gamma) * loss
        return loss_ascent

class FloodingLoss(nn.Module):
    def __init__(self, flooding, criterion):
        super(FloodingLoss, self).__init__()
        self.flooding = flooding
        self.criterion = criterion
        return
    
    def forward(self,output,target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = (loss - self.flooding).abs() + self.flooding
        return loss_ascent


def adjust_learning_rate(optimizer, epoch, args):
    '''set learning rate during the process of pretraining model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < args.tuning_epochs:
        lr = args.lr
    else:
        lr = 0.01
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_loss_value(args, poisoned_data, model_ascent):
    '''Calculate loss value per example
    args:
        Contains default parameters
    poisoned_data:
        the train dataset which contains backdoor data
    model_ascent:
        the model after the process of pretrain
    '''
    # Define loss function
    if args.device == 'cuda':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = torch.utils.data.DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target,_,_,_) in tqdm(enumerate(example_data_loader, start=0)):
        
        img = img.to(args.device)
        target = target.to(args.device)

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in descending order

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record)
    logging.info(f'Top ten loss value: {losses_record_arr[losses_idx[:10]]}')

    return losses_idx

def isolate_data(args, result, losses_idx):
    '''isolate the backdoor data with the calculated loss
    args:
        Contains default parameters
    result:
        the attack result contain the train dataset which contains backdoor data
    losses_idx:
        the index of order about the loss value for each data 
    '''
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    ratio = args.isolation_ratio
    perm = losses_idx[0: int(len(losses_idx) * ratio)]
    permnot = losses_idx[int(len(losses_idx) * ratio):]
    tf_compose = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    train_dataset = result['bd_train'].wrapped_dataset
    data_set_without_tran = train_dataset
    data_set_isolate = result['bd_train']
    data_set_isolate.wrapped_dataset = data_set_without_tran
    data_set_isolate.wrap_img_transform = tf_compose

    data_set_other_without_tran = data_set_without_tran.copy()
    data_set_other = dataset_wrapper_with_transform(
            data_set_other_without_tran,
            tf_compose,
            None,
        )
    # x = result['bd_train']['x']
    # y = result['bd_train']['y']

    data_set_isolate.subset(perm)
    data_set_other.subset(permnot)

    # isolation_examples = list(zip([x[ii] for ii in perm],[y[ii] for ii in perm]))
    # other_examples = list(zip([x[ii] for ii in permnot],[y[ii] for ii in permnot]))
    
    logging.info('Finish collecting {} isolation examples: '.format(len(data_set_isolate)))
    logging.info('Finish collecting {} other examples: '.format(len(data_set_other)))

    return data_set_isolate, data_set_other



def learning_rate_finetuning(optimizer, epoch, args):
    '''set learning rate during the process of finetuing model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate_unlearning(optimizer, epoch, args):
    '''set learning rate during the process of unlearning model 
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    '''
    if epoch < args.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical

def build_gmm(list_mean, list_cov):
    if isinstance(list_mean, list):
        list_mean = torch.stack(list_mean, 0)
    if isinstance(list_cov, list):
        list_cov = torch.stack(list_cov, 0)
    n_components = len(list_mean)
    mvn = MultivariateNormal(list_mean, list_cov)
    # print('batch shape:', mvn.batch_shape)
    # print('event shape:', mvn.event_shape)
    mix = Categorical(torch.ones(n_components))
    gmm = MixtureSameFamily(mix, mvn)

    return gmm

def build_gmm_for_model(args, model, dataloader, criterion, return_features=False):
    record_array = []
    last_linear, hook = register_hook_before_final(model, record_array, store_cpu=True, detach=True, clone=True)

    clean_metrics, \
    clean_test_epoch_predict_list, \
    clean_test_epoch_label_list, \
        = given_dataloader_test(
        model=model,
        test_dataloader=dataloader,
        criterion=criterion,
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=1,
        )
    hook.remove()

    test_acc = clean_metrics["test_acc"]
    print('clean test ACC:', test_acc)

    fet_array = torch.cat(record_array)

    mean_list = []
    cov_list = []    

    for cls in range(args.num_classes):   
        idx = (clean_test_epoch_predict_list == cls)
        fet = fet_array[idx]
        cls_mean = torch.mean(fet, 0)
        cls_cov = torch.cov(fet.T)
        mean_list.append(cls_mean)
        cov_list.append(cls_cov)

    gmm = build_gmm(mean_list, cov_list)

    if return_features:
        return gmm, fet_array
    return gmm


class MLP(nn.Module):
    """ MNIST Encoder from Original Paper's Keras based Implementation.
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """

    def __init__(self, din=2, dout=10, num_filters=32, depth=3, regularize_weight=False):
        super(MLP, self).__init__()
        self.din = din
        self.dout = dout
        self.init_num_filters = num_filters
        self.depth = depth

        self.features = nn.Sequential()

        def _build_linear(din, dout):
            rst = nn.Linear(din,dout)
            if regularize_weight:
                rst = nn.utils.parametrizations.spectral_norm(rst)
            return rst

        for i in range(self.depth):
            if i == 0:
                self.features.add_module('linear%02d' % (i + 1), _build_linear(self.din, self.init_num_filters))
            else:
                self.features.add_module('linear%02d' % (i + 1),
                                        _build_linear(self.init_num_filters, self.init_num_filters))
            self.features.add_module('activation%02d' % (i + 1), nn.LeakyReLU(inplace=True))
            # self.features.add_module('activation%02d' % (i + 1), nn.ReLU(inplace=True))

        self.features.add_module('linear%02d' % (i + 2), _build_linear(self.init_num_filters, self.dout))

    def forward(self, x):
        return self.features(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def reset(self):
        self.features.apply(self.init_weights)


def train_trans_model_V2(args, benign_model, attack_model):
    benign_model.to(args.device, non_blocking=args.non_blocking)
    benign_model.eval()
    attack_model.to(args.device, non_blocking=args.non_blocking)
    attack_model.eval()

    benign_record_array = []
    benign_linear, benign_hook = register_hook_before_final(benign_model, benign_record_array, detach=True, clone=True)
    attack_record_array = []
    attack_linear, attack_hook = register_hook_before_final(attack_model, attack_record_array, detach=True, clone=True)
    benign_hook.remove()
    attack_hook.remove()

    benign_weight = benign_linear.weight.detach().clone()
    attack_weight = attack_linear.weight.detach().clone()
    benign_bias = benign_linear.bias.detach().clone()
    attack_bias = attack_linear.bias.detach().clone()

    dim = benign_weight.shape[-1]

    comp = attack_weight.T @ attack_weight
    factor = torch.matmul(torch.linalg.pinv(comp), attack_weight.T)

    new_weight = factor @ benign_weight
    new_bias = factor @ (benign_bias - attack_bias)

    trans_model = torch.nn.Linear(in_features=dim, out_features=dim, bias=True)
    trans_model.weight.data = new_weight
    trans_model.bias.data = new_bias

    trans_model.to(args.device, non_blocking=args.non_blocking)
    trans_model.eval()

    return trans_model


def train_trans_model(args, benign_model, attack_model, dataloader):
    benign_model.to(args.device, non_blocking=args.non_blocking)
    benign_model.eval()
    attack_model.to(args.device, non_blocking=args.non_blocking)
    attack_model.eval()

    benign_record_array = []
    _, benign_hook = register_hook_before_final(benign_model, benign_record_array, detach=True, clone=True)
    attack_record_array = []
    _, attack_hook = register_hook_before_final(attack_model, attack_record_array, detach=True, clone=True)

    epochs = 10

    def _build_trans_model(fet_dim, epochs, args):
        trans_model = MLP(din=fet_dim, dout=fet_dim, num_filters=fet_dim, depth=1)
        optimizer = torch.optim.Adam(trans_model.parameters(), lr=1e-3, betas=(0.8,0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        trans_model.to(args.device, non_blocking=args.non_blocking)
        trans_model.train()
        return trans_model, optimizer, scheduler

    fet_dim = None
    trans_model, optimizer, scheduler = None, None, None
    pbar = tqdm(range(epochs))
    for _ in pbar:
        for batch_idx, (x, target, *additional_info) in enumerate(dataloader):
            x = x.to(args.device, non_blocking=args.non_blocking)

            with torch.no_grad():
                benign_model(x)
                attack_model(x)

                benign_fet = benign_record_array[-1]
                attack_fet = attack_record_array[-1]

                benign_record_array.clear()
                attack_record_array.clear()

            if fet_dim is None:
                fet_dim = benign_fet.shape[-1]
                trans_model, optimizer, scheduler = _build_trans_model(fet_dim, epochs, args)

            predic_fet = trans_model(benign_fet)
            loss = F.mse_loss(predic_fet, attack_fet)

            lv = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        pbar.set_description(f"last loss: {lv:.4f}")
        pbar.refresh()

    benign_hook.remove()
    attack_hook.remove()

    return trans_model



def calc_metrics(args, benign_model, attack_model, dataloader, criterion):

    clean_metrics, \
    clean_test_epoch_predict_list, \
    clean_test_epoch_label_list, \
    clean_test_epoch_logits_list \
        = given_dataloader_test(
        model=benign_model,
        test_dataloader=dataloader,
        criterion=criterion,
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=2,
        )

    attack_metrics, \
    attack_test_epoch_predict_list, \
    attack_test_epoch_label_list, \
    attack_test_epoch_logits_list \
        = given_dataloader_test(
        model=attack_model,
        test_dataloader=dataloader,
        criterion=criterion,
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=2,
        )
    
    clean_acc = all_acc(clean_test_epoch_predict_list, clean_test_epoch_label_list)
    attack_acc = all_acc(attack_test_epoch_predict_list, attack_test_epoch_label_list)
    print('clean acc:', clean_acc)
    print('attack acc:', attack_acc)
    
    clean_test_epoch_logits_list.to(args.device)
    attack_test_epoch_logits_list.to(args.device)
    p = F.softmax(clean_test_epoch_logits_list, dim=-1)
    q = F.softmax(attack_test_epoch_logits_list, dim=-1)
    logp = F.log_softmax(clean_test_epoch_logits_list, dim=-1)
    logq = F.log_softmax(attack_test_epoch_logits_list, dim=-1)
    KL_pq = torch.mean(p*(logp-logq))
    KL_qp = torch.mean(q*(logq-logp))
    print('KL clean||attack:', KL_pq, torch.sqrt(0.5*KL_pq))
    print('KL attack||clean:', KL_qp, torch.sqrt(0.5*KL_qp))

    Hellinger_dis = torch.sqrt(1-torch.mean(torch.sum(torch.sqrt(p*q), axis=-1)))
    print('Helliger distance:', Hellinger_dis, np.sqrt(2)*Hellinger_dis)


def calc_wasserstein_metrics_GAN(args, benign_model, attack_model, dataloader):

    trans_model = train_trans_model_V2(args, benign_model, attack_model)

    benign_record_array = []
    benign_linear, benign_hook = register_hook_before_final(benign_model, benign_record_array, detach=True, clone=True)
    attack_record_array = []
    attack_linear, attack_hook = register_hook_before_final(attack_model, attack_record_array, detach=True, clone=True)
    data_shape = [args.input_channel, args.input_width, args.input_height]

    random_dataset = RandomDataset(data_shape=data_shape, 
                                    len=10000, 
                                    random_type='normal', 
                                    num_classes=args.num_classes,
                                    fully_random=True,
                                    )
    random_dataloader = DataLoader(random_dataset, batch_size=args.batch_size, shuffle=False,
                                    drop_last=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers, )
    


    def _build_gsw_model(fet_dim, epochs, args):
        gsw_model = MLP(din=fet_dim, dout=fet_dim, num_filters=128, depth=2, regularize_weight=True)
        # optimizer = torch.optim.Adam(gsw_model.parameters(), lr=2e-4, betas=(0.5,0.999))
        optimizer = torch.optim.Adam(gsw_model.parameters(), lr=1e-3, betas=(0.8, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        gsw_model.to(args.device, non_blocking=args.non_blocking)
        gsw_model.train()
        return gsw_model, optimizer, scheduler
    
    def _wasserstein_loss(X, Y, model):
        Xslices = model(X)
        Yslices = model(Y)

        dist = torch.mean(Xslices) - torch.mean(Yslices)
        return dist
    
    def _wasserstein1_dist(X, Y, model):
        Xslices = model(X)
        Yslices = model(Y)
        Xslices_sorted = torch.sort(Xslices, dim=0)[0]
        Yslices_sorted = torch.sort(Yslices, dim=0)[0]

        dist = torch.mean(torch.abs(Xslices_sorted-Yslices_sorted))
        return dist

    trans_optimizer = torch.optim.Adam(trans_model.parameters(), lr=2e-4, betas=(0.5,0.999))

    epochs = 100
    fet_dim = None
    gsw_model, optimizer, scheduler = None, None, None
    for e in range(epochs):
        lv_list = []
        for batch_idx, (x, target, *additional_info) in enumerate(random_dataloader):
            x = x.to(args.device, non_blocking=args.non_blocking)

            with torch.no_grad():
                benign_model(x)
                attack_model(x)

                benign_fet = benign_record_array[-1]
                attack_fet = attack_record_array[-1]

                benign_record_array.clear()
                attack_record_array.clear()

                trans_fet = trans_model(benign_fet)

            if fet_dim is None:
                fet_dim = benign_fet.shape[-1]
                gsw_model, optimizer, scheduler = _build_gsw_model(fet_dim, epochs, args)

            optimizer.zero_grad()
            loss = -1 * _wasserstein_loss(trans_fet, attack_fet, gsw_model)

            loss.backward()
            optimizer.step()

            lv = loss.item()
            lv_list.append(lv)

            if batch_idx%10 != 0:
                continue
            
            trans_optimizer.zero_grad()
            trans_fet = trans_model(benign_fet)
            loss_trans = torch.mean(gsw_model(trans_fet))

            loss_trans.backward()
            trans_optimizer.step()
        
        print(f'epoch {e}:', np.mean(lv_list))

    gsw_model.eval()
    dist_list = []
    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(random_dataloader):
            x = x.to(args.device, non_blocking=args.non_blocking)

            benign_model(x)
            attack_model(x)

            benign_fet = benign_record_array[-1]
            attack_fet = attack_record_array[-1]

            benign_record_array.clear()
            attack_record_array.clear()

            trans_fet = trans_model(benign_fet)
            dist = _wasserstein1_dist(trans_fet, attack_fet, gsw_model)
            dist_list.append(dist.item())


    benign_hook.remove()
    attack_hook.remove()

    w1_dist = np.mean(dist_list)
    print('W1 dist:', w1_dist)
    
    return w1_dist



def calc_wasserstein_metrics(args, benign_model, attack_model, dataloader):

    trans_model = train_trans_model_V2(args, benign_model, attack_model)

    benign_record_array = []
    benign_linear, benign_hook = register_hook_before_final(benign_model, benign_record_array, detach=True, clone=True)
    attack_record_array = []
    attack_linear, attack_hook = register_hook_before_final(attack_model, attack_record_array, detach=True, clone=True)
    data_shape = [args.input_channel, args.input_width, args.input_height]

    random_dataset = RandomDataset(data_shape=data_shape, 
                                    len=10000, 
                                    random_type='normal', 
                                    num_classes=args.num_classes,
                                    fully_random=True,
                                    )
    random_dataloader = DataLoader(random_dataset, batch_size=args.batch_size, shuffle=False,
                                    drop_last=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers, )
    


    def _build_gsw_model(fet_dim, epochs, args):
        gsw_model = MLP(din=fet_dim, dout=fet_dim, num_filters=128, depth=2, regularize_weight=True)
        optimizer = torch.optim.Adam(gsw_model.parameters(), lr=1e-3, betas=(0.8,0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        gsw_model.to(args.device, non_blocking=args.non_blocking)
        gsw_model.train()
        return gsw_model, optimizer, scheduler
    
    def _wasserstein_loss(X, Y, model):
        Xslices = model(X)
        Yslices = model(Y)

        dist = torch.mean(Xslices) - torch.mean(Yslices)
        return dist
    
    def _wasserstein1_dist(X, Y, model):
        Xslices = model(X)
        Yslices = model(Y)
        Xslices_sorted = torch.sort(Xslices, dim=0)[0]
        Yslices_sorted = torch.sort(Yslices, dim=0)[0]

        dist = torch.mean(torch.abs(Xslices_sorted-Yslices_sorted))
        return dist


    epochs = 10
    fet_dim = None
    gsw_model, optimizer, scheduler = None, None, None
    for e in range(epochs):
        lv_list = []
        for batch_idx, (x, target, *additional_info) in enumerate(random_dataloader):
            x = x.to(args.device, non_blocking=args.non_blocking)

            with torch.no_grad():
                benign_model(x)
                attack_model(x)

                benign_fet = benign_record_array[-1]
                attack_fet = attack_record_array[-1]

                benign_record_array.clear()
                attack_record_array.clear()

                trans_fet = trans_model(benign_fet)

            if fet_dim is None:
                fet_dim = benign_fet.shape[-1]
                gsw_model, optimizer, scheduler = _build_gsw_model(fet_dim, epochs, args)

            loss = -1 * _wasserstein_loss(trans_fet, attack_fet, gsw_model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            lv = loss.item()
            lv_list.append(lv)
        
        print(f'epoch {e}:', lv_list[-1])

    gsw_model.eval()
    dist_list = []
    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(random_dataloader):
            x = x.to(args.device, non_blocking=args.non_blocking)

            benign_model(x)
            attack_model(x)

            benign_fet = benign_record_array[-1]
            attack_fet = attack_record_array[-1]

            benign_record_array.clear()
            attack_record_array.clear()

            trans_fet = trans_model(benign_fet)
            dist = _wasserstein1_dist(trans_fet, attack_fet, gsw_model)
            dist_list.append(dist.item())


    benign_hook.remove()
    attack_hook.remove()

    w1_dist = np.mean(dist_list)
    print('W1 dist:', w1_dist)
    
    return w1_dist


class TSA(defense):
    r"""Anti-backdoor learning: Training clean models on poisoned data.
    
    basic structure: 
    
    1. config args, save_path, fix random seed
    2. load the backdoor attack data and backdoor test data
    3. abl defense:
        a. pre-train model
        b. isolate the special data(loss is low) as backdoor data
        c. unlearn the backdoor data and learn the remaining data
    4. test the result and get ASR, ACC, RC 
       
    .. code-block:: python
    
        parser = argparse.ArgumentParser(description=sys.argv[0])
        abl.add_arguments(parser)
        args = parser.parse_args()
        abl_method = abl(args)
        if "result_file" not in args.__dict__:
            args.result_file = 'one_epochs_debug_badnet_attack'
        elif args.result_file is None:
            args.result_file = 'one_epochs_debug_badnet_attack'
        result = abl_method.defense(args.result_file)
    
    .. Note::
        @article{li2021anti,
            title={Anti-backdoor learning: Training clean models on poisoned data},
            author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={14900--14912},
            year={2021}
            }

    Args:
        baisc args: in the base class
        tuning_epochs (int): number of the first tuning epochs to run
        finetuning_ascent_model (bool): whether finetuning model after sperate the poisoned data
        finetuning_epochs (int): number of the finetuning epochs to run
        unlearning_epochs (int): number of the unlearning epochs to run
        lr_finetuning_init (float): initial finetuning learning rate
        lr_unlearning_init (float): initial unlearning learning rate
        momentum (float): momentum of sgd during the process of finetuning and unlearning
        weight_decay (float): weight decay of sgd during the process of finetuning and unlearning
        isolation_ratio (float): ratio of isolation data from the whole poisoned data
        gradient_ascent_type (str): type of gradient ascent (LGA, Flooding)
        gamma (float): value of gamma for LGA
        flooding (float): value of flooding for Flooding
        
    """ 
    

    def __init__(self,args):
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--benign_folder', type=str, help='path to folder containing reference model')

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--interval', type=int, help='frequency of save model')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        #set the parameter for the abl defense
        parser.add_argument('--tuning_epochs', type=int, help='number of tune epochs to run')
        parser.add_argument('--finetuning_ascent_model', type=bool, help='whether finetuning model')
        parser.add_argument('--finetuning_epochs', type=int, help='number of finetuning epochs to run')
        parser.add_argument('--unlearning_epochs', type=int, help='number of unlearning epochs to run')
        parser.add_argument('--lr_finetuning_init', type=float, help='initial finetuning learning rate')
        parser.add_argument('--lr_unlearning_init', type=float, help='initial unlearning learning rate')
        parser.add_argument('--momentum', type=float, help='momentum')
        parser.add_argument('--weight_decay', type=float, help='weight decay')
        parser.add_argument('--isolation_ratio', type=float, help='ratio of isolation data')
        parser.add_argument('--gradient_ascent_type', type=str, help='type of gradient ascent')
        parser.add_argument('--gamma', type=float, help='value of gamma')
        parser.add_argument('--flooding', type=float, help='value of flooding')

        return parser
        

    def get_attack_result(self, attack_folder):
        attack_path = os.path.join('record', attack_folder)
        result = load_attack_result(os.path.join(attack_path,'attack_result.pt'), return_backdoor_model=True)
        return result

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def mitigation(self):
        result = self.result 
        ###a. pre-train model
        poisoned_data, model_ascent = self.pre_train(args,result)
        
        ###b. isolate the special data(loss is low) as backdoor data
        losses_idx = compute_loss_value(args, poisoned_data, model_ascent)
        logging.info('----------- Collect isolation data -----------')
        isolation_examples, other_examples = isolate_data(args, result, losses_idx)

        ###c. unlearn the backdoor data and learn the remaining data
        model_new = self.train_unlearning(args,result,model_ascent,isolation_examples,other_examples)

        result = {}
        result['model'] = model_new
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model_new.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self):
        args = self.args
        attack_result = self.get_attack_result(args.attack_folder)
        benign_result = self.get_attack_result(args.benign_folder)
        attack_model = attack_result['model']
        benign_model = benign_result['model']

        criterion = argparser_criterion(args)

        bd_test_dataset_with_transform = attack_result['bd_test']
        if bd_test_dataset_with_transform is not None:
            bd_test_dataloader = DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                            drop_last=False,
                                            pin_memory=args.pin_memory, num_workers=args.num_workers, )

            bd_metrics, \
            bd_test_epoch_predict_list, \
            bd_test_epoch_label_list, \
            bd_test_epoch_original_index_list, \
            bd_test_epoch_poison_indicator_list, \
            bd_test_epoch_original_targets_list = test_given_dataloader_on_mix(
                model=attack_model,
                test_dataloader=bd_test_dataloader,
                criterion=criterion,
                non_blocking=args.non_blocking,
                device=args.device,
                verbose=1,
            )

            bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
            test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

            print('test ASR:', test_asr)


        clean_test_dataset_with_transform = benign_result['clean_test']
        clean_test_dataloader = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False,
                                        pin_memory=args.pin_memory, num_workers=args.num_workers, )
        
        calc_metrics(args, benign_model, attack_model, clean_test_dataloader, criterion)

        data_shape = [args.input_channel, args.input_width, args.input_height]
        random_dataset = RandomDataset(data_shape=data_shape, 
                                        len=len(clean_test_dataset_with_transform), 
                                        random_type='normal', 
                                        num_classes=args.num_classes,
                                        fully_random=False,
                                      )

        random_dataloader = DataLoader(random_dataset, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False,
                                        pin_memory=args.pin_memory, num_workers=args.num_workers, )
        calc_metrics(args, benign_model, attack_model, random_dataloader, criterion)


        calc_wasserstein_metrics(args, benign_model, attack_model, clean_test_dataloader)
        exit(0)
 

        
        benign_gmm, benign_fets = build_gmm_for_model(args, benign_model, clean_test_dataloader, criterion, return_features=True)
        attack_gmm, attack_fets = build_gmm_for_model(args, attack_model, clean_test_dataloader, criterion, return_features=True)
        print(benign_gmm)
        print(attack_gmm)

        # trns_B_to_A = train_trans_model(args, benign_model, attack_model, clean_test_dataloader)

        # n_samples = 10000
        # X = benign_gmm.sample((n_samples,))
        # print(X.shape)

        # tX = trns_B_to_A(X.to(args.device))

        X = benign_fets
        tX = attack_fets
        print(X.shape)
        print(tX.shape)

        benign_log_probs = benign_gmm.log_prob(X)
        print(benign_log_probs.shape)
        print(torch.mean(benign_log_probs))

        attack_log_probs = attack_gmm.log_prob(tX)
        print(attack_log_probs.shape)
        print(torch.mean(attack_log_probs))

        eKL = torch.mean(benign_log_probs) - torch.mean(attack_log_probs)
        print(eKL)

        exit(0)
 

        print(clean_test_epoch_label_list.shape)

        weight_mat = last_linear.weight.data.cpu()
        bias = last_linear.bias.data.cpu()

        for cls in range(args.num_classes):
            idx = (clean_test_epoch_predict_list == cls)
            fet = fet_array[idx]
            cls_mean = torch.mean(fet, 0)
            cls_cov = torch.cov(fet.T)
            print(cls, cls_mean.shape, cls_cov.shape)

            cls_mean = torch.unsqueeze(cls_mean, 1)
            cls_cov_inv = torch.linalg.pinv(cls_cov)

            w = cls_cov_inv @ cls_mean # in [:, 1]
            print(cls_cov_inv.shape, w.shape)

            wf = w.T
            of = weight_mat[cls:cls+1,:]

            print(wf.shape, of.shape)

            print('d2:', torch.norm(wf-of, p=2))
            print('d2 after regularized:', torch.norm(wf/torch.norm(wf)-of/torch.norm(of), p=2))
            print('mean factor:', torch.mean(wf/of))

            cos = torch.nn.functional.cosine_similarity(wf, of)
            rad_per_degree = np.pi/180
            print('cos:', cos.item(), 'degree:', torch.acos(cos).item()/rad_per_degree)

            bhat = cls_mean.T @ cls_mean
            print('bhat:', bhat, 'b:', bias[cls])

            v = torch.rand(wf.shape)
            vcos = torch.nn.functional.cosine_similarity(v, of)
            print(v.shape, vcos, torch.acos(vcos).item()/rad_per_degree)

            u = cls_mean.T
            ucos = torch.nn.functional.cosine_similarity(u, of)
            print(u.shape, ucos, torch.acos(ucos).item()/rad_per_degree)

            print('d2:', torch.norm(u-of, p=2))
            print('d2 after regularized:', torch.norm(u/torch.norm(u)-of/torch.norm(of), p=2))
            print('mean factor:', torch.mean(u/of))
            print('===' * 20)
 

        hook.remove()
        exit(0)

        result = self.mitigation()
        return result

    def pre_train(self, args, result):
        '''Pretrain the model with raw data
        args:
            Contains default parameters
        result:
            attack result(details can be found in utils)
        '''
        agg = Metric_Aggregator()
        # Load models
        logging.info('----------- Network Initialization --------------')
        model_ascent = generate_cls_model(args.model,args.num_classes)
        if "," in self.device:
            model_ascent = torch.nn.DataParallel(
                model_ascent,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model_ascent.device_ids[0]}'
            model_ascent.to(self.args.device)
        else:
            model_ascent.to(self.args.device)
        logging.info('finished model init...')
        # initialize optimizer 
        # because the optimizer has parameter nesterov
        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        # define loss functions
        # recommend to use cross entropy
        criterion = argparser_criterion(args).to(args.device)
        if args.gradient_ascent_type == 'LGA':
            criterion = LGALoss(args.gamma,criterion).to(args.device)
        elif args.gradient_ascent_type == 'Flooding':
            criterion = FloodingLoss(args.flooding,criterion).to(args.device)
        else:
            raise NotImplementedError

        logging.info('----------- Data Initialization --------------')

        # tf_compose = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        tf_compose = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        train_dataset = result['bd_train'].wrapped_dataset
        data_set_without_tran = train_dataset
        data_set_o = result['bd_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = tf_compose
        
        # data_set_isolate = result['bd_train']
        # data_set_isolate.wrapped_dataset = data_set_without_tran
        # data_set_isolate.wrap_img_transform = tf_compose

        # # data_set_other = copy.deepcopy(data_set_isolate)
        # # x = result['bd_train']['x']
        # # y = result['bd_train']['y']
        # losses_idx = range(50000)
        # ratio = args.isolation_ratio
        # perm = losses_idx[0: int(len(losses_idx) * ratio)]
        # permnot = losses_idx[int(len(losses_idx) * ratio):]
        # data_set_isolate.subset(perm)
        # data_set_o.subset(permnot)
        # data_set_other = copy.deepcopy(data_set_o)
        poisoned_data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)    

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        train_loss_list = []
        train_mix_acc_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        logging.info('----------- Train Initialization --------------')
        for epoch in range(0, args.tuning_epochs):
            logging.info("Epoch {}:".format(epoch + 1))
            adjust_learning_rate(optimizer, epoch, args)
            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            train_clean_acc, \
            train_asr, \
            train_ra = self.train_step(args, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)  

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model_ascent,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": epoch,

                "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                "train_acc": train_mix_acc,
                "train_acc_clean_only": train_clean_acc,
                "train_asr_bd_only": train_asr,
                "train_ra_bd_only": train_ra,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Train Acc": train_mix_acc_list,
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}pre_train_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Train Loss": train_loss_list,
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}pre_train_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}pre_train_df.csv")

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                state_dict = {
                    "model": model_ascent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_current": epoch,
                }
                torch.save(state_dict, args.checkpoint_save + "pre_train_state_dict.pt")

        agg.summary().to_csv(f"{args.save_path}pre_train_df_summary.csv")

        return data_set_o, model_ascent

    def train_unlearning(self, args, result, model_ascent, isolate_poisoned_data, isolate_other_data):
        '''train the model with remaining data and unlearn the backdoor data
        args:
            Contains default parameters
        result:
            attack result(details can be found in utils)
        model_ascent:
            the model after pretrain
        isolate_poisoned_data:
            the dataset of 'backdoor' data
        isolate_other_data:
            the dataset of remaining data
        '''
        agg = Metric_Aggregator()
        # Load models
        ### TODO: load model from checkpoint
        # logging.info('----------- Network Initialization --------------')
        # if "," in args.device:
        #     model_ascent = torch.nn.DataParallel(
        #         model_ascent,
        #         device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
        #     )
        # else:
        #     model_ascent.to(args.device)
        # model_ascent.to(args.device)
        logging.info('Finish loading ascent model...')
        # initialize optimizer
        # Because nesterov we do not use other optimizer
        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        # define loss functions
        # you can use other criterion, but the paper use cross validation to unlearn sample
        if args.device == 'cuda':
            criterion = argparser_criterion(args).cuda()
        else:
            criterion = argparser_criterion(args)
        
        tf_compose_finetuning = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
        tf_compose_unlearning = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    
        isolate_poisoned_data.wrap_img_transform = tf_compose_finetuning
        isolate_poisoned_data_loader = torch.utils.data.DataLoader(dataset=isolate_poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        )

        isolate_other_data.wrap_img_transform = tf_compose_unlearning
        isolate_other_data_loader = torch.utils.data.DataLoader(dataset=isolate_other_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                )

        test_tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
        data_bd_testset = result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        train_loss_list = []
        train_mix_acc_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        logging.info('----------- Train Initialization --------------')

        if args.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            logging.info('----------- Finetuning isolation model --------------')
            for epoch in range(0, args.finetuning_epochs):
                learning_rate_finetuning(optimizer, epoch, args)
                train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra = self.train_step(args, isolate_other_data_loader, model_ascent, optimizer, criterion, epoch + 1)  

                clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra = self.eval_step(
                    model_ascent,
                    data_clean_loader,
                    data_bd_loader,
                    args,
                )

                agg({
                    "epoch": epoch,

                    "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                    "train_acc": train_mix_acc,
                    "train_acc_clean_only": train_clean_acc,
                    "train_asr_bd_only": train_asr,
                    "train_ra_bd_only": train_ra,

                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })

                train_loss_list.append(train_epoch_loss_avg_over_batch)
                train_mix_acc_list.append(train_mix_acc)
                train_clean_acc_list.append(train_clean_acc)
                train_asr_list.append(train_asr)
                train_ra_list.append(train_ra)

                clean_test_loss_list.append(clean_test_loss_avg_over_batch)
                bd_test_loss_list.append(bd_test_loss_avg_over_batch)
                ra_test_loss_list.append(ra_test_loss_avg_over_batch)
                test_acc_list.append(test_acc)
                test_asr_list.append(test_asr)
                test_ra_list.append(test_ra)

                general_plot_for_epoch(
                    {
                        "Train Acc": train_mix_acc_list,
                        "Test C-Acc": test_acc_list,
                        "Test ASR": test_asr_list,
                        "Test RA": test_ra_list,
                    },
                    save_path=f"{args.save_path}finetune_acc_like_metric_plots.png",
                    ylabel="percentage",
                )

                general_plot_for_epoch(
                    {
                        "Train Loss": train_loss_list,
                        "Test Clean Loss": clean_test_loss_list,
                        "Test Backdoor Loss": bd_test_loss_list,
                        "Test RA Loss": ra_test_loss_list,
                    },
                    save_path=f"{args.save_path}finetune_loss_metric_plots.png",
                    ylabel="percentage",
                )

                agg.to_dataframe().to_csv(f"{args.save_path}finetune_df.csv")

                if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                    state_dict = {
                        "model": model_ascent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch_current": epoch,
                    }
                    torch.save(state_dict, args.checkpoint_save + "finetune_state_dict.pt")
        agg.summary().to_csv(f"{args.save_path}finetune_df_summary.csv")


        best_acc = 0
        best_asr = 0
        logging.info('----------- Model unlearning --------------')
        for epoch in range(0, args.unlearning_epochs):
            
            learning_rate_unlearning(optimizer, epoch, args)
            train_epoch_loss_avg_over_batch, \
            train_mix_acc, \
            train_clean_acc, \
            train_asr, \
            train_ra = self.train_step_unlearn(args, isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)  

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model_ascent,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": epoch,

                "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                "train_acc": train_mix_acc,
                "train_acc_clean_only": train_clean_acc,
                "train_asr_bd_only": train_asr,
                "train_ra_bd_only": train_ra,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Train Acc": train_mix_acc_list,
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}unlearn_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Train Loss": train_loss_list,
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}unlearn_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}unlearn_df.csv")

            if args.frequency_save != 0 and epoch % args.frequency_save == args.frequency_save - 1:
                state_dict = {
                    "model": model_ascent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_current": epoch,
                }
                torch.save(state_dict, args.checkpoint_save + "unlearn_state_dict.pt")
        
        agg.summary().to_csv(f"{args.save_path}unlearn_df_summary.csv")
        agg.summary().to_csv(f"{args.save_path}abl_df_summary.csv")
        return model_ascent

    
    def train_step(self, args, train_loader, model_ascent, optimizer, criterion, epoch):
        '''Pretrain the model with raw data for each step
        args:
            Contains default parameters
        train_loader:
            the dataloader of train data
        model_ascent:
            the initial model
        optimizer:
            optimizer during the pretrain process
        criterion:
            criterion during the pretrain process
        epoch:
            current epoch
        '''
        losses = 0
        size = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        model_ascent.train()

        for idx, (img, target, original_index, poison_indicator, original_targets) in enumerate(train_loader, start=1):
            
            img = img.to(args.device)
            target = target.to(args.device)

            pred = model_ascent(img)
            loss_ascent = criterion(pred,target)

            losses += loss_ascent * img.size(0)
            size += img.size(0)
            optimizer.zero_grad()
            loss_ascent.backward()
            optimizer.step()

            batch_loss_list.append(loss_ascent.item())
            batch_predict_list.append(torch.max(pred, -1)[1].detach().clone().cpu())
            batch_label_list.append(target.detach().clone().cpu())
            batch_original_index_list.append(original_index.detach().clone().cpu())
            batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())

        train_epoch_loss_avg_over_batch, \
        train_epoch_predict_list, \
        train_epoch_label_list, \
        train_epoch_poison_indicator_list, \
        train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                            torch.cat(batch_predict_list), \
                                            torch.cat(batch_label_list), \
                                            torch.cat(batch_poison_indicator_list), \
                                            torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )

        return train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra

    def train_step_unlearn(self, args, train_loader, model_ascent, optimizer, criterion, epoch):
        '''Pretrain the model with raw data for each step
        args:
            Contains default parameters
        train_loader:
            the dataloader of train data
        model_ascent:
            the initial model
        optimizer:
            optimizer during the pretrain process
        criterion:
            criterion during the pretrain process
        epoch:
            current epoch
        '''
        losses = 0
        size = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        model_ascent.train()

        for idx, (img, target, original_index, poison_indicator, original_targets) in enumerate(train_loader, start=1):
            
            img = img.to(args.device)
            target = target.to(args.device)

            pred = model_ascent(img)
            loss_ascent = criterion(pred,target)

            losses += loss_ascent * img.size(0)
            size += img.size(0)
            optimizer.zero_grad()
            (-loss_ascent).backward()
            optimizer.step()

            batch_loss_list.append(loss_ascent.item())
            batch_predict_list.append(torch.max(pred, -1)[1].detach().clone().cpu())
            batch_label_list.append(target.detach().clone().cpu())
            batch_original_index_list.append(original_index.detach().clone().cpu())
            batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())

        train_epoch_loss_avg_over_batch, \
        train_epoch_predict_list, \
        train_epoch_label_list, \
        train_epoch_poison_indicator_list, \
        train_epoch_original_targets_list = sum(batch_loss_list) / len(batch_loss_list), \
                                            torch.cat(batch_predict_list), \
                                            torch.cat(batch_label_list), \
                                            torch.cat(batch_poison_indicator_list), \
                                            torch.cat(batch_original_targets_list)

        train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )

        return train_epoch_loss_avg_over_batch, \
                train_mix_acc, \
                train_clean_acc, \
                train_asr, \
                train_ra

    def eval_step(
            self,
            netC,
            clean_test_dataloader,
            bd_test_dataloader,
            args,
    ):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = TSA.add_base_arguments(parser)
    parser = TSA.add_arguments(parser)
    args = parser.parse_args()
    TSA.add_yaml_to_args(args)
    args = TSA.process_args(args)
    tsa_method = TSA(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    tsa_method.prepare(args)
    result = tsa_method.defense()