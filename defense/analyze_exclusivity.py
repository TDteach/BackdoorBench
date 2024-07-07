'''
BELT: Old-School Backdoor Attacks can Evade the State-of-the-Art Defense with Backdoor Exclusivity Lifting
This file is modified based on the following source:
link : https://github.com/JSun20220909/BELT/blob/main/BELT/exclusivity_calculator.py
The defense method is called AE.
@article{qiu2023belt,
  title={BELT: Old-School Backdoor Attacks can Evade the State-of-the-Art Defense with Backdoor Exclusivity Lifting},
  author={Qiu, Huming and Sun, Junjie and Zhang, Mi and Pan, Xudong and Yang, Min},
  journal={arXiv preprint arXiv:2312.04902},
  year={2023}
}
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.optim as optim
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
from utils.useful_tools import get_clip_image


class Perturbations(nn.Module):
    def __init__(self, args):
        super(Perturbations, self).__init__()

        self.args = args
        self.mask = torch.Tensor(data.mask).permute(2, 0, 1).cuda()
        self.pattern = torch.Tensor(data.pattern / 255.).permute(2, 0, 1).cuda()


        self.trigger = self.pattern * self.mask
        self.max_radius = ((1 - self.pattern.round()) - self.pattern) * self.mask
        self.Lambda = 0.1

        upper_perturbations = torch.zeros([args.input_channel, args.input_height, args.input_width])
        self.upper_perturbations = upper_perturbations.cuda().requires_grad_(True)

        self.ce = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([self.upper_perturbations], lr=args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.clip_func = get_clip_image(args)

    def dynamic(self, outputs_upper, targets_poison):
        norm_pre = F.normalize(outputs_upper[0], dim=-1)
        targets_pre = norm_pre[targets_poison].item()
        other_max_pre = norm_pre[F.one_hot(targets_poison[0], num_classes=self.args.num_classes) == 0].max(dim=-1)[0].item()
        change = targets_pre - other_max_pre
        self.Lambda = np.maximum(self.Lambda + change, 0.)

    def add_trigger(self, inputs, upper_perturbations=None):
        if upper_perturbations is None:
            upper_perturbations = self.upper_perturbations
        inputs_poison_upper = (1 - self.mask) * inputs + self.pattern * self.mask + upper_perturbations
        inputs_poison_upper = self.clip_func(inputs_poison_upper)
        return inputs_poison_upper

    def loss(self, outputs_upper, targets_poison, epoch):
        if epoch >= self.args.epochs // 2:
            self.dynamic(outputs_upper, targets_poison)
        loss_perturbations = self.ce(outputs_upper, targets_poison) + self.Lambda * torch.norm(self.max_radius - self.upper_perturbations)
        loss = loss_perturbations
        return loss

    def robustness_index(self, inputs, inputs_poison_upper):
        upper_perturbations = inputs_poison_upper[0] - (inputs[0] * (1 - self.mask) + self.pattern * self.mask)
        upper_radius = torch.norm(upper_perturbations)
        spec = 1 - upper_radius.item() / torch.norm(self.max_radius).item()
        spec *= 100
        return spec


class Trainer(nn.Module):
    def __init__(self, model, args):
        super(Trainer, self).__init__()
        self.model = model
        self.args = args

    def train_step(self, index, perturbations, inputs, targets_poison):
        args = self.args
        net = self.model
        net.eval()

        best_spec = 100
        max_upper_perturbations = 0
        pbar = tqdm(range(1, args.epochs+1))
        for epoch in pbar:
            inputs_poison_upper = perturbations.add_trigger(inputs)

            perturbations.optimizer.zero_grad()
            outputs_upper, _ = net(inputs_poison_upper)
            loss = perturbations.loss(outputs_upper, targets_poison, epoch)
            loss.backward()
            perturbations.optimizer.step()

            with torch.no_grad():
                perturbations.upper_perturbations.clamp_(-perturbations.trigger, perturbations.mask - perturbations.trigger)

            train_loss = loss.item()
            _, predicted = outputs_upper.max(1)
            asr = predicted.eq(targets_poison).item()

            spec = perturbations.robustness_index(inputs, inputs_poison_upper)
            if asr:
                best_spec = spec
                max_upper_perturbations = perturbations.upper_perturbations.clone()
                logs = '{} - Epoch: [{}][{}/{}]\t Loss: {:.4f}\t SPE: {:.4f}%\t upper_perturbations: {:.4f}\t Lambda: {:.4f}'
                pbar.set_description(logs.format('TRAIN', index, epoch, epochs, train_loss, spec,
                                                 torch.norm(perturbations.upper_perturbations).item(), perturbations.Lambda))

        return best_spec, max_upper_perturbations

    def forward(self, clean_loader, poison_loader):
        device = self.device
        all_spec = []
        for index, ((inputs, targets, _), (inputs_poison, targets_poison, _)) in enumerate(zip(clean_loader, poison_loader)):
            if index >= 100:
                break
            inputs, targets = inputs.to(device), targets.to(device).long()
            inputs_poison, targets_poison = inputs_poison.to(device), targets_poison.to(device).long()

            perturbations = Perturbations()
            best_spec, max_upper_perturbations = self.train_step(index, perturbations, inputs, targets_poison)

            all_spec.append(best_spec)

            # inputs_poison_upper = perturbations.add_trigger(inputs, upper_perturbations=max_upper_perturbations)
            # image_array = (inputs_poison_upper[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # plt.imsave('debug_perturb_ori_after0.png', image_array)

        all_spec = np.array(all_spec)
        print(all_spec.min(), all_spec.max(), all_spec.mean())




class AE(defense):
    def __init__(self,args):
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
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
        attack_model = attack_result['model']

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


        clean_test_dataset_with_transform = attack_result['clean_test']
        clean_test_dataloader = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False,
                                        pin_memory=args.pin_memory, num_workers=args.num_workers, )
        
        d
        
        # result = self.mitigation()
        # return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = AE.add_base_arguments(parser)
    parser = AE.add_arguments(parser)
    args = parser.parse_args()
    AE.add_yaml_to_args(args)
    args = AE.process_args(args)
    ae_method = AE(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    ae_method.prepare(args)
    result = ae_method.defense()