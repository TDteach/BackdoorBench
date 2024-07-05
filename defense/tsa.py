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
        pass

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


        
        # benign_gmm, benign_fets = build_gmm_for_model(args, benign_model, clean_test_dataloader, criterion, return_features=True)
        # attack_gmm, attack_fets = build_gmm_for_model(args, attack_model, clean_test_dataloader, criterion, return_features=True)
        # print(benign_gmm)
        # print(attack_gmm)

        # # trns_B_to_A = train_trans_model(args, benign_model, attack_model, clean_test_dataloader)

        # # n_samples = 10000
        # # X = benign_gmm.sample((n_samples,))
        # # print(X.shape)

        # # tX = trns_B_to_A(X.to(args.device))

        # X = benign_fets
        # tX = attack_fets

        # benign_log_probs = benign_gmm.log_prob(X)
        # attack_log_probs = attack_gmm.log_prob(tX)

        # eKL = torch.mean(benign_log_probs) - torch.mean(attack_log_probs)
        # print('KL between two GMMs:', eKL)





        # record_array = []
        # last_linear, hook = register_hook_before_final(benign_model, record_array, store_cpu=True, detach=True, clone=True)

        # clean_metrics, \
        # clean_test_epoch_predict_list, \
        # clean_test_epoch_label_list, \
        #     = given_dataloader_test(
        #     model=benign_model,
        #     test_dataloader=clean_test_dataloader,
        #     criterion=criterion,
        #     non_blocking=args.non_blocking,
        #     device=args.device,
        #     verbose=1,
        #     )
        # hook.remove()

        # fet_array = torch.cat(record_array)

        # weight_mat = last_linear.weight.data.cpu()
        # bias = last_linear.bias.data.cpu()

        # for cls in range(args.num_classes):
        #     idx = (clean_test_epoch_predict_list == cls)
        #     fet = fet_array[idx]
        #     cls_mean = torch.mean(fet, 0)
        #     cls_cov = torch.cov(fet.T)

        #     cls_mean = torch.unsqueeze(cls_mean, 1)
        #     cls_cov_inv = torch.linalg.pinv(cls_cov)

        #     w = cls_cov_inv @ cls_mean # in [:, 1]

        #     wf = w.T
        #     of = weight_mat[cls:cls+1,:]

        #     print(wf.shape, of.shape)

        #     print('d2:', torch.norm(wf-of, p=2))
        #     print('d2 after regularized:', torch.norm(wf/torch.norm(wf)-of/torch.norm(of), p=2))
        #     print('mean factor:', torch.mean(wf/of))

        #     cos = torch.nn.functional.cosine_similarity(wf, of)
        #     rad_per_degree = np.pi/180
        #     print('cos:', cos.item(), 'degree:', torch.acos(cos).item()/rad_per_degree)

        #     bhat = cls_mean.T @ cls_mean
        #     print('bhat:', bhat, 'b:', bias[cls])

        #     v = torch.rand(wf.shape)
        #     vcos = torch.nn.functional.cosine_similarity(v, of)
        #     print(v.shape, vcos, torch.acos(vcos).item()/rad_per_degree)

        #     u = cls_mean.T
        #     ucos = torch.nn.functional.cosine_similarity(u, of)
        #     print(u.shape, ucos, torch.acos(ucos).item()/rad_per_degree)

        #     print('d2:', torch.norm(u-of, p=2))
        #     print('d2 after regularized:', torch.norm(u/torch.norm(u)-of/torch.norm(of), p=2))
        #     print('mean factor:', torch.mean(u/of))
        #     print('===' * 20)
 

        # hook.remove()
        # exit(0)

        # result = self.mitigation()
        # return result

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