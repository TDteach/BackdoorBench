'''
Exploring the Orthogonality and Linearity of Backdoor Attacks
the code is modifed from 
https://github.com/KaiyuanZh/OrthogLinearBackdoor/tree/main
The defense method is called TED.
@inproceedings{zhang2024exploring,
  title={Exploring the Orthogonality and Linearity of Backdoor Attacks},
  author={Zhang, Kaiyuan and Cheng, Siyuan and Shen, Guangyu and Tao, Guanhong and An, Shengwei and Makur, Anuran and Ma, Shiqing and Zhang, Xiangyu},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={225--225},
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

import logging
from defense.base import defense

from torch.utils.data import DataLoader

from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from utils.save_load_attack import load_attack_result, save_defense_result

from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import auc, roc_curve, confusion_matrix
from pyod.models.pca import PCA

import shap
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import time



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



############################################################################
# Customized functions
############################################################################
def sub_network(model, network):
    if network.startswith('resnet'):
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'SequentialWithArgs':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(2, torch.nn.ReLU())
        children.insert(-1, torch.nn.AvgPool2d(4))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['BasicBlock', 'BatchNorm2d']
    elif network.startswith('preactresnet'):
        children = list(model.children())
        nchildren = []
        for c in children:
            # print(c.__class__.__name__)
            if c.__class__.__name__ == 'Sequential':
                nchildren += list(c.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['PreActBlock', 'BatchNorm2d', 'PreActBottleneck']

    elif network == 'wrn':
        children = list(model.children())
        nchildren = []
        for c in children:
            if c.__class__.__name__ == 'NetworkBlock':
                nchildren += list(c.layer.children())
            else:
                nchildren.append(c)
        children = nchildren
        children.insert(-1, torch.nn.AvgPool2d(8))
        children.insert(-1, torch.nn.Flatten())
        target_layers = ['BasicBlock', 'Conv2d']
    else:
        raise NotImplementedError
    
    # Find the target layers
    target_ids = []
    for i, c in enumerate(children):
        if c.__class__.__name__ in target_layers:
            target_ids.append(i)

    return children, target_ids


def split_model(children, target_id):
    model_head = torch.nn.Sequential(*children[:target_id])
    model_tail = torch.nn.Sequential(*children[target_id:])
    return model_head, model_tail

class Custom_model(nn.Module):
    def __init__(self, model):
        super(Custom_model, self).__init__()
        self.model = model

    def forward(self, x):
        for layer in self.model.children():
            if layer.__class__.__name__ == 'Flatten':
                x = x.view(x.size(0), -1)
            else:
                x = layer(x)
        return x
    
def compute_all_layer_gradients(model, inputs, labels, preprocess):
    model.zero_grad()
    output = model(preprocess(inputs))

    # if args.attack == 'composite':
    #     CLASS_A = 0
    #     CLASS_B = 1
    #     CLASS_C = 2  # A + B -> C
    #     criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive', device=DEVICE)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(output, labels)
    loss.backward()

    gradients = []

    for name, p in model.named_parameters():
        if 'conv' in name:
            grad = p.grad.clone().abs().detach()
            # print(f"gradients shape: {grad.shape}")
            gradients.append(grad.cpu().view(-1))
    gradients = torch.cat(gradients)
    return gradients


class OL(defense):

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
        self.result = result
        return result

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def mitigation(self):
        pass

    def eval_linear(self, target_label, preprocess=None, mode='attack'):
        args = self.args
        device = args.device
        model = self.result['model']
        clean_testset = self.result['clean_test']
        poison_testset = self.result['bd_test']
        model_name = self.result['model_name']

        if preprocess is None:
            preprocess = lambda x:x

        test_loader = torch.utils.data.DataLoader(clean_testset, batch_size=args.batch_size, shuffle=True)
        if mode=='clean':
            poison_loader = torch.utils.data.DataLoader(clean_testset, batch_size=args.batch_size, shuffle=True)
        else:
            poison_loader = torch.utils.data.DataLoader(poison_testset, batch_size=args.batch_size, shuffle=True)

        children, target_ids = sub_network(model, model_name)


        linearity_scores = list()
        for target_id in target_ids:
            time_start = time.time()

            model_head, model_tail = split_model(children, target_id)

            model_head = Custom_model(model_head).to(device)
            model_tail = Custom_model(model_tail).to(device)

            # Get the output of the target layer
            with torch.no_grad():
                # Get data of clean images
                for _, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    # Inputs for the tail model (Only 1 batch)
                    background = model_head(preprocess(x_batch))
                    break
                
                # Get the output of the tail model
                outputs = model_tail(background)
                pred = outputs.max(dim=1)[1]
                acc = (pred == y_batch).sum().item() / x_batch.size(0)

                # Get data of poisoned images
                for _, (x_batch, y_batch, *other_info) in enumerate(poison_loader):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    # Inputs for the tail model (Only 1 batch)
                    data = model_head(preprocess(x_batch))
                    break
                
                # Get the output of the tail model
                outputs = model_tail(data)
                pred = outputs.max(dim=1)[1]
                asr = (pred == y_batch).sum().item() / x_batch.size(0)

            print(f'Current target layer: {target_id}, Accuracy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')

            # Use SHAP to identify the important neurons
            background = background[:16]
            explainer = shap.DeepExplainer(model_tail, background)

            # Calculate the SHAP values for the test set
            shap_values = explainer.shap_values(data)[target_label].reshape(data.shape[0], data.shape[1], -1)
            shap_values = np.max(shap_values, axis=2)
            shap_values = shap_values.mean(axis=0)

            # TODO: Select the top-k% neurons
            _k = 0.03
            n_select = int(np.ceil(shap_values.size * _k))
            selected_neurons = np.argsort(shap_values)[-n_select:]

            time_end = time.time()
            # print(f'Selected neurons: {selected_neurons}, time: {time_end - time_start}')

            n_neurons = len(selected_neurons)
            test_acti = data[:, selected_neurons].reshape(data.shape[0], n_neurons, -1).max(dim=2)[0]
            n_activated = ((test_acti > 1e-3).sum(dim=0) / test_acti.shape[0]) > 0.9
            n_activated = n_activated.sum().item()

            # Mutate the values of the selected neurons
            neuron_mask = torch.zeros_like(data)
            neuron_mask[:, selected_neurons] = 1

            layer_mean = data.mean(dim=[0], keepdim=True)

            linear_inputs = np.arange(0, 3, 0.1)
            # linear_inputs = np.arange(0, 1, 0.1)
            linear_outputs = []
            for w in linear_inputs:
                mute = w * layer_mean * neuron_mask
                data_mute = data + mute
                output = model_tail(data_mute)

                if w == 0:
                    base = output
                else:
                    diff = (output - base)[:, target_label].detach().cpu().numpy()
                    linear_outputs.append(diff)

            linear_inputs = np.array(linear_inputs)[1:].reshape(-1, 1)
            linear_outputs = np.array(linear_outputs)

            # Measure the linearity of the mapping
            reg = LinearRegression().fit(linear_inputs, linear_outputs)
            r2 = r2_score(linear_outputs, reg.predict(linear_inputs))
            print(f'No. activated: {n_activated} ({n_neurons}), Linearity score: {r2}')
            print()

            linearity_scores.append(r2)

        print('min:', np.min(linearity_scores), 'at', np.argmin(linearity_scores))
        print('max:', np.max(linearity_scores), 'at', np.argmax(linearity_scores))
        print('mean:', np.mean(linearity_scores))


    def eval_orthogonal(self, preprocess=None, mode='attack'):
        args = self.args
        device = args.device
        model = self.result['model']
        clean_testset = self.result['clean_test']
        poison_testset = self.result['bd_test']

        model.eval()
        model.to(device)

        if preprocess is None:
            preprocess = lambda x:x

        test_loader = torch.utils.data.DataLoader(clean_testset, batch_size=args.batch_size, shuffle=True)
        if mode=='clean':
            poison_loader = torch.utils.data.DataLoader(clean_testset, batch_size=args.batch_size, shuffle=True)
        else:
            poison_loader = torch.utils.data.DataLoader(poison_testset, batch_size=args.batch_size, shuffle=True)


        # Calculate the gradient for clean and poisoned images
        clean_gradients = []
        poison_gradients = []

        n_batch = 8
        for i in range(n_batch):
            x_clean, y_clean = next(iter(test_loader))
            x_poison, y_poison, *other_infor = next(iter(poison_loader))

            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
            x_poison, y_poison = x_poison.to(device), y_poison.to(device)

            batch_clean_grad = compute_all_layer_gradients(model, x_clean, y_clean, preprocess)
            batch_poison_grad = compute_all_layer_gradients(model, x_poison, y_poison, preprocess)

            # print(f"batch_clean_grad shape: {batch_clean_grad.shape}, batch_poison_grad shape: {batch_poison_grad.shape}")

            clean_gradients.append(batch_clean_grad)
            poison_gradients.append(batch_poison_grad)
        
        clean_gradients = torch.mean(torch.stack(clean_gradients), dim=0)
        poison_gradients = torch.mean(torch.stack(poison_gradients), dim=0)

        # print(f"clean_grad shape: {clean_gradients.shape}, poison_grad shape: {poison_gradients.shape}")

        cosine_similarity = torch.nn.functional.cosine_similarity(clean_gradients, poison_gradients, dim=0)
        angle = torch.acos(cosine_similarity) * 180 / np.pi
        print("=====================================")
        print(f"cosine_similarity {cosine_similarity.item()}, angle {angle.item()}")



    def defense(self):
        self.get_attack_result(self.args.attack_folder)
        self.eval_orthogonal()
        print('==='*20)
        self.eval_linear(target_label=0)


        # result = self.mitigation()
        # return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = OL.add_base_arguments(parser)
    parser = OL.add_arguments(parser)
    args = parser.parse_args()
    OL.add_yaml_to_args(args)
    args = OL.process_args(args)
    method = OL(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    method.prepare(args)
    result = method.defense()