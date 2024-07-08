'''
Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection

This file is modified based on the following source:
link : https://github.com/TDteach/Demon-in-the-Variant/blob/master/pysrc/SCAn.py
The detection method is called SCAn.
@inproceedings{tang2021demon,
    title={Demon in the variant: Statistical analysis of $\{$DNNs$\}$ for robust backdoor contamination detection},
    author={Tang, Di and Wang, XiaoFeng and Tang, Haixu and Zhang, Kehuan},
    booktitle={30th USENIX Security Symposium (USENIX Security 21)},
    pages={1541--1558},
    year={2021}}

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. SCAn detection:
        a. Leverage the target model to generate representations for all input images.
        b. Estimate the parameters by running an EM algorithm.
        c. calculate the identity vector and decompose the representations.
        d. estimate the parameters for the mixture model.
        e. perform the likelihood ratio test.
    4. compute TPR and FPR

'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from defense.base import defense
import scipy
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.nCHW_nHWC import *

import tqdm
import heapq
from PIL import Image
from utils.bd_dataset_v2 import dataset_wrapper_with_transform,xy_iter, prepro_cls_DatasetBD_v2
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from collections import Counter
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import csv
from sklearn import metrics

def get_features_labels(args, model, target_layer, data_loader):

    def feature_hook(module, input_, output_):
        global feature_vector
        feature_vector = output_
        return None

    h = target_layer.register_forward_hook(feature_hook)

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, *other_info) in enumerate(data_loader):
            global feature_vector
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            if len(feature_vector.shape) > 2:
                feature_vector = torch.sum(torch.flatten(feature_vector, 2), 2)
            # feature_vector = torch.flatten(feature_vector, 1)
            current_feature = feature_vector.detach().cpu().numpy()
            current_labels = targets.cpu().numpy()

            # Store features
            features.append(current_feature)
            labels.append(current_labels)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    h.remove()  # Rmove the hook

    return features, labels

EPS = 1e-4
class SCAn:
    def __init__(self, args):
        self.args = args
        pass

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        cls_labels = list(lc_model.keys())
        cls_labels.sort()
        y = []
        for c in cls_labels:
            y.append(lc_model[c]['score'])
        y = np.asarray(y)
        ai = self.calc_anomaly_index(y / np.max(y))
        ai_dict = {c: v for c,v in zip(cls_labels, ai)}
        return ai_dict

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index
    
    def calc_matrix_inverse(self, ma):
        # ma_numpy = ma.cpu().numpy()
        # rst_numpy = np.linalg.pinv(ma_numpy)
        # rst = torch.from_numpy(rst_numpy).to(self.args.device)
        rst = torch.linalg.pinv(ma) # it's strange that torch is much slower than numpy in linalg.pinv
        return rst.to(torch.float64)
    
    def build_global_model(self, reprs, labels, n_classes):
        reprs = torch.from_numpy(reprs).to(self.args.device)
        labels = torch.from_numpy(labels).to(self.args.device)
        reprs = reprs.to(torch.float64)
        labels = labels.to(torch.long)

        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = torch.mean(reprs, dim=0)
        X = reprs - mean_a

        cnt_L = torch.zeros(L, device=self.args.device, dtype=torch.long)
        mean_f = torch.zeros([L, M], device=self.args.device, dtype=torch.float64)
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = torch.sum(idx)
            mean_f[k] = torch.mean(X[idx], dim=0)

        u = torch.zeros([N, M], device=self.args.device, dtype=torch.float64)
        e = torch.zeros([N, M], device=self.args.device, dtype=torch.float64)
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = torch.cov(u.T)
        Se = torch.cov(e.T)

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        time_list = []
        loss_list = []
        min_loss_loc = None
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            st_time = time.perf_counter()
            n_iters += 1
            last_Su = Su
            last_Se = Se

            # F = torch.linalg.pinv(Se)
            F = self.calc_matrix_inverse(Se)
            SuF = torch.matmul(Su, F)

            G_set = list()
            for k in range(L):
                # G = -torch.linalg.pinv(cnt_L[k] * Su + Se)
                G = -self.calc_matrix_inverse(cnt_L[k] * Su + Se)
                G = torch.matmul(G, SuF)
                G_set.append(G)

            u_m = torch.zeros([L, M], device=self.args.device, dtype=torch.float64)
            e = torch.zeros([N, M], device=self.args.device, dtype=torch.float64)
            u = torch.zeros([N, M], device=self.args.device, dtype=torch.float64)

            for i in range(N):
                vec = X[i:i+1]
                k = labels[i]
                G = G_set[k]
                dd = torch.matmul(torch.matmul(Se, G), vec.T)
                u_m[k] = u_m[k] - dd.T

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = torch.cov(u.T)
            Se = torch.cov(e.T)

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = torch.norm(dif_Su)
            dist_Se = torch.norm(dif_Se)
            ed_time = time.perf_counter()
            time_list.append(ed_time-st_time)
            # print(n_iters, time_list[-1], np.mean(time_list))
            print(f'iter {n_iters}:', dist_Su.item(), dist_Se.item(), f'using {time_list[-1]} seconds')

            loss = (dist_Su+dist_Se).item()
            if min_loss_loc is None or loss < loss_list[min_loss_loc]:
                min_loss_loc = len(loss_list)
            loss_list.append(loss)
            if len(loss_list)-min_loss_loc-1 > 5:
                break

        gb_model = {
            'Su': Su,
            'Se': Se,
            'F': self.calc_matrix_inverse(Se),
            'mean': mean_f,
            'mean_a': mean_a,
        }
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model):
        reprs = torch.from_numpy(reprs).to(self.args.device)
        labels = torch.from_numpy(labels).to(self.args.device)
        reprs = reprs.to(torch.float64)
        labels = labels.to(torch.long)

        Su = gb_model['Su']
        Se = gb_model['Se']
        F = gb_model['F']

        # F = torch.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]

        cls_labels = np.unique(labels.cpu().numpy())
        cls_labels.sort()
        L = len(cls_labels)

        # mean_a = torch.mean(reprs, dim=0)
        mean_a = gb_model['mean_a']
        X = reprs - mean_a

        lc_model = dict()
        for k in cls_labels:
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)

            _rst = {
                'subg': subg,
                'u1': i_u1.view(-1),
                'u2': i_u2.view(-1),
                'score': i_sc[0][0].item(),
                'num': torch.sum(selected_idx).item(),
            }
            print('[class-%d] outlier_original_score = %f, calculated on %d samples' % (k, _rst['score'], _rst['num']) )

            lc_model[k] = _rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = torch.rand(N, device=self.args.device, dtype=torch.float64)

        if (N == 1):
            subg[0] = 0
            return (subg, X.clone(), X.clone())

        if torch.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if torch.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -torch.ones(N, device=self.args.device, dtype=torch.float64)

        # EM
        steps = 0
        while (torch.norm(subg - last_z1) > EPS) and (torch.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.clone()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (torch.sum(idx1) == 0) or (torch.sum(idx2) == 0):
                break
            if torch.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = torch.mean(X[idx1], dim=0)
            if torch.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = torch.mean(X[idx2], dim=0)

            u1 = u1.view([1,-1])
            u2 = u2.view([1,-1])
            bias = torch.matmul(torch.matmul(u1, F), u1.T) - torch.matmul(torch.matmul(u2, F), u2.T)
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i:i+1]
                delta = torch.matmul(torch.matmul(e1, F), e2.T)
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        # G = -torch.linalg.pinv(N * Su + Se)
        G = -self.calc_matrix_inverse(N * Su + Se)
        mu = torch.zeros([1, M], device=self.args.device, dtype=torch.float64)
        SeG = torch.matmul(Se,G)
        for i in range(N):
            vec = X[i:i+1]
            dd = torch.matmul(SeG, vec.T)
            mu = mu - dd.T

        b1 = torch.matmul(torch.matmul(mu, F), mu.T) - torch.matmul(torch.matmul(u1, F), u1.T)
        b2 = torch.matmul(torch.matmul(mu, F), mu.T) - torch.matmul(torch.matmul(u2, F), u2.T)
        n1 = torch.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i:i+1]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * torch.matmul(torch.matmul(e1, F), e2.T)

        return sc / N



class scan(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
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

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/scan/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--num_samples_per_class', type=int)
        parser.add_argument('--target_layer', type=str)

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/scan_pretrain/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'detection_info/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
                
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model = model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        self.device = self.args.device
    
    def cal(self, true, pred):
        TN, FP, FN, TP = confusion_matrix(true, pred).ravel()
        return TN, FP, FN, TP 
    def metrix(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = TP/(TP+FP)
        acc = (TP+TN)/(TN+FP+FN+TP)
        return TPR, FPR, precision, acc
    def filtering(self):
        start = time.perf_counter()
        self.set_devices()
        fix_random(self.args.random_seed)

        ### a. load model, bd train data and transforms
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
            model.eval()
        else:
            model.to(self.args.device)
            model.eval()
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)

        module_dict = dict(model.named_modules())
        try:
            target_layer = module_dict[args.target_layer]
        except:
            logging.error(f'has not found target_layer in following modules:')
            logging.error(module_dict.keys())
            raise KeyError


        def _get_images_for_classes(_dataset, hard_limit=False):
            images = []
            labels = []
            for img, label in _dataset:
                images.append(img)
                labels.append(label)
            class_idx_whole = []
            # num = int(self.args.clean_sample_num / self.args.num_classes)
            num = int(self.args.num_samples_per_class)
            if num == 0:
                num = 1
            for i in range(self.args.num_classes):
                _ind = np.where(np.array(labels)==i)[0]
                if hard_limit and len(_ind) < num:
                    raise f"class {i} contains only {len(_ind)} samples, which is less then {num} required"
                elif not hard_limit:
                    _num = min(num, len(_ind))
                else:
                    _num = num
                class_idx_whole.append(np.random.choice(_ind, _num, replace=False))
            class_idx_whole = np.concatenate(class_idx_whole, axis=0)
            image_c = [images[i] for i in class_idx_whole]
            label_c = [labels[i] for i in class_idx_whole]
            return image_c, label_c

        ### b. find a clean sample from test dataset
        clean_test_dataset = self.result['clean_test'].wrapped_dataset
        image_c, label_c = _get_images_for_classes(clean_test_dataset, hard_limit=False)
        clean_dataset = xy_iter(image_c, label_c,transform=test_tran)
        clean_dataloader = DataLoader(clean_dataset, self.args.batch_size, shuffle=True)
        clean_features,clean_labels = get_features_labels(args, model, target_layer, clean_dataloader)
        logging.info('clean_features has been collected')

        ### bb. find a clean sample from train dataset
        clean_train_dataset = self.result['clean_train'].wrapped_dataset
        image_c, label_c = _get_images_for_classes(clean_train_dataset, hard_limit=True)
        clean_train_dataset = xy_iter(image_c, label_c,transform=test_tran)
        clean_train_dataloader = DataLoader(clean_train_dataset, self.args.batch_size, shuffle=True)
        clean_train_features,clean_train_labels = get_features_labels(args, model, target_layer, clean_train_dataloader)
        logging.info('clean_train_features has been collected')


        ### c. load training dataset with poison samples
        bd_test_dataset = self.result['bd_test'].wrapped_dataset
        # pindex = np.where(np.array(bd_test_dataset.poison_indicator) == 1)[0]
        images_poison = []
        labels_poison = []
        class_cnt = np.zeros(self.args.num_classes,dtype=int)
        for idx, (img, label, original_index, poison_or_not, original_target) in enumerate(bd_test_dataset):
            if label == original_target: continue
            images_poison.append(img)
            labels_poison.append(label)
            class_cnt[label] += 1
            if np.max(class_cnt) >= args.num_samples_per_class:
                break
        attack_target = np.argmax(class_cnt)
        if class_cnt[attack_target] < args.num_samples_per_class:
            raise f"not enough poison samples provided: class {attack_target} contains only {class_cnt[attack_target]} samples < {args.num_samples_per_class} required"
        labels_poison = np.asarray(labels_poison)
        attack_target_indicator = (labels_poison == attack_target)
        labels_poison = labels_poison[attack_target_indicator]
        _new_images_poison = list()
        for img, ind in zip(images_poison, attack_target_indicator):
            if ind:
                _new_images_poison.append(img)
        images_poison = _new_images_poison

        ### d. get features of poison dataset
        poison_dataset = xy_iter(images_poison, labels_poison,transform=test_tran)
        poison_dataloader = DataLoader(poison_dataset, self.args.batch_size, shuffle=False)
        poison_features, poison_labels = get_features_labels(args, model, target_layer, poison_dataloader)
        logging.info('poison_features has been collected')
        

        ### e. build global model
        feats_clean = np.array(clean_features)
        class_indices_clean = np.array(clean_labels)

        scan = SCAn(self.args)
        gb_model = scan.build_global_model(feats_clean, class_indices_clean, self.args.num_classes)


        ### f. build local model
        feats_inspection = np.array(clean_train_features)
        class_indices_inspection = np.array(clean_train_labels)
        clean_lc_model = scan.build_local_model(feats_inspection, class_indices_inspection, gb_model)

        ind = class_indices_inspection == attack_target
        target_class_feats = feats_inspection[ind]
        target_class_labels = class_indices_inspection[ind]

        threshold = np.exp(2)
        test_np_list = [0.01, 0.02, 0.05, 0.1]
        for npos in test_np_list:
            if npos < 1:
                npos = int(npos*self.args.num_samples_per_class)
            np_ratio = npos/self.args.num_samples_per_class* 100
            print('==='*20)
            print(f'when class target class {attack_target} continas {npos} = {np_ratio:.2f}% poison samples:')

            new_feats = np.concatenate([poison_features[:npos], target_class_feats[npos:]], axis=0)
            new_labels = np.concatenate([poison_labels[:npos], target_class_labels[npos:]], axis=0)

            tgt_lc_model = scan.build_local_model(new_feats, new_labels, gb_model)
            cp_lc_model = copy.deepcopy(clean_lc_model)
            cp_lc_model[attack_target] = tgt_lc_model[attack_target]
            score = scan.calc_final_score(cp_lc_model)

            for c in range(self.args.num_classes):
                if score[c] > threshold:
                    print('[class-%d] outlier_score = %f, is detected as infected' % (c, score[c]) )
                else:
                    print('[class-%d] outlier_score = %f' % (c, score[c]) )
            

        exit(0)

        suspicious_indices = []
        flag_list = []

        for target_class in range(args.num_classes):

            print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

            if score[target_class] <= threshold: 
                continue
            flag_list.append([target_class, score[target_class]])
            tar_label = (class_indices_all == target_class)
            all_label = np.arange(len(class_indices_all))
            tar = all_label[tar_label]

            cluster_0_indices = []
            cluster_1_indices = []

            cluster_0_clean = []
            cluster_1_clean = []

            for index, i in enumerate(lc_model[target_class]['subg']):
                if i == 1:
                    if tar[index] > size_inspection_set:
                        cluster_1_clean.append(tar[index])
                    else:
                        cluster_1_indices.append(tar[index])
                else:
                    if tar[index] > size_inspection_set:
                        cluster_0_clean.append(tar[index])
                    else:
                        cluster_0_indices.append(tar[index])


            if len(cluster_0_clean) < len(cluster_1_clean): # if most clean samples are in cluster 1
                suspicious_indices += cluster_0_indices
            else:
                suspicious_indices += cluster_1_indices
                
        true_index = np.zeros(len(images_poison))
        for i in range(len(true_index)):
            if i in pindex:
                true_index[i] = 1
        if len(suspicious_indices)==0:
            tn = len(true_index) - np.sum(true_index)
            fp = np.sum(true_index)
            fn = 0
            tp = 0
            f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
            csv_write.writerow([args.result_file, tn,fp,fn,tp, 0,0, 'None'])
            f.close()
        else: 
            logging.info("Flagged label list: {}".format(",".join(["{}: {}".format(y_label, s) for y_label, s in flag_list])))
            findex = np.zeros(len(images_poison))
            for i in range(len(findex)):
                if i in suspicious_indices:
                    findex[i] = 1
            if np.sum(findex) == 0:
                tn = len(true_index) - np.sum(true_index)
                fp = np.sum(true_index)
                fn = 0
                tp = 0
            else:
                tn, fp, fn, tp = self.cal(true_index, findex)
            TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)

            new_TP = tp
            new_FN = fn*9
            new_FP = fp*1
            precision = new_TP / (new_TP + new_FP) if new_TP + new_FP != 0 else 0
            recall = new_TP / (new_TP + new_FN) if new_TP + new_FN != 0 else 0
            fw1 = 2*(precision * recall)/ (precision + recall) if precision + recall != 0 else 0
            end = time.perf_counter()
            time_miniute = (end-start)/60

            f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
            csv_write = csv.writer(f)
            csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'target'])
            csv_write.writerow([args.result_file, tn, fp, fn, tp, TPR, FPR, [i for i,j in flag_list]])
            f.close()


    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    scan.add_arguments(parser)
    args = parser.parse_args()
    scan_method = scan(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = scan_method.detection(args.result_file)
