'''
Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics
the code is modifed from 
https://github.com/tedbackdoordefense/ted
The defense method is called TED.
@inproceedings{mo2024robust,
  title={Robust backdoor detection for deep learning via topological evolution dynamics},
  author={Mo, Xiaoxing and Zhang, Yechao and Zhang, Leo Yu and Luo, Wei and Sun, Nan and Hu, Shengshan and Gao, Shang and Xiang, Yang},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={171--171},
  year={2024},
  organization={IEEE Computer Society}
}
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

from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import auc, roc_curve, confusion_matrix
from pyod.models.pca import PCA



def get_activation_hook(activation_record, name):
    def hook(model, input, output):
        if name not in activation_record:
            activation_record[name] = list()
        activation_record[name].append(output.detach().cpu().clone())
    return hook

def register_hooks(model, activation_record):
    hook_handles = []
    net_children = model.modules()
    layer_names = []

    index = 0
    for _, child in enumerate(net_children):
        name = None
        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            name = "Conv2d_" + str(index)
        elif isinstance(child, nn.ReLU):
            name = "Relu_" + str(index)
        elif isinstance(child, nn.Linear):
            name = "Linear_" + str(index)

        if name is not None:
            hook_handles.append(
                child.register_forward_hook(get_activation_hook(activation_record, name))
            )
            index += 1
            layer_names.append(name)

    return hook_handles, layer_names



def get_rankings(model, dataloader, criterion, reference_activations=None, reference_predicts=None):
    activation_record = dict()
    hook_handles, layer_names = register_hooks(model, activation_record)

    metrics, \
    predict_list, \
    label_list, \
        = given_dataloader_test(
        model=model,
        test_dataloader=dataloader,
        criterion=criterion,
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=1,
        )
    
    if reference_predicts is None:
        ref_pred = predict_list
        st = 1
    else:
        ref_pred = reference_predicts
        st = 0

    # get ranking for each layer
    rankings = []
    for ln in layer_names:
        z = torch.cat(activation_record[ln])
        z = torch.reshape(z, (z.shape[0], -1))
        activation_record[ln] = z

        zn2 = torch.sum(torch.square(z), dim=1,keepdim=True) 
        if reference_activations is None:
            dist2 = zn2 + zn2.T - 2* z @ z.T
        else:
            w = reference_activations[ln]
            wn2 = torch.sum(torch.square(w), dim=1, keepdim=True)
            dist2 = zn2 + wn2.T - 2 * z @ w.T

        rk = []
        _, order = torch.sort(dist2, dim=1)
        for i in range(order.shape[0]):
            for j in range(st, order.shape[1]):
                if ref_pred[order[i,j]] == predict_list[i]:
                    break
            rk.append(j)
        rk = np.asarray(rk)-st
        rankings.append(rk)
    rankings = np.asarray(rankings).T

    for h in hook_handles:
        h.remove()

    return rankings, activation_record, predict_list



class TED(defense):

    def __init__(self,args):
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        
        parser.add_argument('--reference_size', type=int, default=400, help='the number of clean reference images')
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
        num_classes = args.num_classes
        criterion = argparser_criterion(args)

        attack_result = self.get_attack_result(args.attack_folder)
        attack_model = attack_result['model']
        attack_model.eval()
        attack_model.to(args.device)

        clean_test_dataset_with_transform = attack_result['clean_test']

        n_clean = len(clean_test_dataset_with_transform)
        assert n_clean >= args.reference_size, "not enough reference images in the test dataset"

        ref_index = []
        cls_n = args.reference_size // num_classes
        cls_needed = np.ones(num_classes, dtype=int) * cls_n
        for i in range(len(clean_test_dataset_with_transform)):
            _, lb = clean_test_dataset_with_transform[i]
            if cls_needed[lb] > 0:
                ref_index.append(i)
                cls_needed[lb] -= 1
        for cls in range(num_classes):
            if cls_needed[cls] > 0:
                raise f"in test dataset, class {cls} contains only {cls_n-cls_needed[cls]} images, but {cls_n} images are required"
        reference_dataset = torch.utils.data.Subset(clean_test_dataset_with_transform, ref_index)
        logging.info(f"real number of reference images is {len(reference_dataset)}")

        reference_loader = DataLoader(reference_dataset, batch_size=args.batch_size, shuffle=False,
                                    drop_last=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers, )


        reference_rankings, reference_activations, reference_predicts = get_rankings(attack_model, reference_loader, criterion)

        print(reference_rankings.shape)
        print(reference_rankings)
        print(np.mean(np.sum(reference_rankings,axis=1)))

        bd_test_dataset_with_transform = attack_result['bd_test']
        if len(bd_test_dataset_with_transform) > len(reference_dataset):
            bd_index = np.random.choice(len(bd_test_dataset_with_transform), len(reference_dataset), replace=False) 
            bd_dataset = torch.utils.data.Subset(bd_test_dataset_with_transform, bd_index)
        else:
            bd_dataset = bd_test_dataset_with_transform
        bd_loader = DataLoader(bd_dataset, batch_size=args.batch_size, shuffle=False,
                                    drop_last=False,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers, )
        bd_rankings, bd_activations, bd_predicts = get_rankings(attack_model, bd_loader, criterion, reference_activations, reference_predicts)

        print(bd_rankings.shape)
        print(bd_rankings)
        print(np.mean(np.sum(bd_rankings,axis=1)))

        pca = PCA(contamination=0.01, n_components='mle')
        pca.fit(reference_rankings)

        n_ref, n_bd = len(reference_rankings), len(bd_rankings)
        all_labels = np.concatenate([np.zeros(n_ref), np.ones(n_bd)], axis=0)
        all_rankings = np.concatenate([reference_rankings, bd_rankings], axis=0)
        all_scores = pca.decision_function(all_rankings)
        all_preds = pca.predict(all_rankings)

        fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
        print("AUC:", auc(fpr, tpr))

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        print("TPR:", tp / (tp + fn))
        print("True Positives (TP):", tp)
        print("False Positives (FP):", fp)
        print("True Negatives (TN):", tn)
        print("False Negatives (FN):", fn)

        # result = self.mitigation()
        # return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = TED.add_base_arguments(parser)
    parser = TED.add_arguments(parser)
    args = parser.parse_args()
    TED.add_yaml_to_args(args)
    args = TED.process_args(args)
    tsa_method = TED(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    tsa_method.prepare(args)
    result = tsa_method.defense()