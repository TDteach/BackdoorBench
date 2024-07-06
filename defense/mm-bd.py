'''
MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic
the code is modified based on 
https://github.com/wanghangpsu/MM-BD/blob/main/univ_bd.py
The defense method is called MM_BD.
@inproceedings{wang2023mm,
  title={Mm-bd: Post-training detection of backdoor attacks with arbitrary backdoor pattern types using a maximum margin statistic},
  author={Wang, Hang and Xiang, Zhen and Miller, David J and Kesidis, George},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={15--15},
  year={2023},
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

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, all_acc, general_plot_for_epoch, given_dataloader_test
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.useful_tools import get_clip_image

from scipy.stats import median_abs_deviation as MAD
from scipy.stats import gamma

class MM_BD(defense):

    def __init__(self,args):
        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        
        parser.add_argument('--num_starts', type=int, help='the number of different images initiated to find the max-margin') # default =40
        parser.add_argument('--optimize_steps', type=int, help='the number of gradient ascent to find the max-margin') # default = 300
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
        device = args.device
        num_classes = args.num_classes
        NSTEP = args.optimize_steps
        criterion = argparser_criterion(args)

        attack_result = self.get_attack_result(args.attack_folder)
        attack_model = attack_result['model']
        attack_model.to(device)
        attack_model.eval()

        clip_image = get_clip_image(args)

        res = []
        for t in range(num_classes):
            images = torch.rand([args.num_starts, args.input_channel, args.input_width, args.input_height]).to(args.device)
            images.requires_grad = True

            last_loss = 1000
            labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
            onehot_label = F.one_hot(labels, num_classes=num_classes)

            optimizer, scheduler = argparser_opt_scheduler(None, args, param=[images])

            for iter_idx in range(NSTEP):

                optimizer.zero_grad()
                outputs = attack_model(clip_image(images))

                loss = -1 * torch.sum((outputs * onehot_label)) \
                    + torch.sum(torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
                loss.backward(retain_graph=True)
                optimizer.step()
                if abs(last_loss - loss.item())/abs(last_loss)< 1e-5:
                    break
                last_loss = loss.item()

            res.append(torch.max(torch.sum((outputs * onehot_label), dim=1)\
                    - torch.max((1-onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
            print(t, res[-1])


        stats = res
        mad = MAD(stats, scale='normal')
        abs_deviation = np.abs(stats - np.median(stats))
        score = abs_deviation / mad
        print(score)

        # np.save('results.npy', np.array(res))
        ind_max = np.argmax(stats)
        r_eval = np.amax(stats)
        r_null = np.delete(stats, ind_max)

        shape, loc, scale = gamma.fit(r_null)
        pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null)+1)
        print(pv)
        if pv > 0.05:
            print('No Attack!')
        else:
            print('There is attack with target class {}'.format(np.argmax(stats)))


        # result = self.mitigation()
        # return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = MM_BD.add_base_arguments(parser)
    parser = MM_BD.add_arguments(parser)
    args = parser.parse_args()
    MM_BD.add_yaml_to_args(args)
    args = MM_BD.process_args(args)
    mmbd_method = MM_BD(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    mmbd_method.prepare(args)
    result = mmbd_method.defense()