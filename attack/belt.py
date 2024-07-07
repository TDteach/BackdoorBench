'''
BELT: Old-School Backdoor Attacks can Evade the State-of-the-Art Defense with Backdoor Exclusivity Lifting
the code is modified based on 
https://github.com/JSun20220909/BELT/blob/main/BELT/BadNet_BELT.py
this script is for belt attack
@article{qiu2023belt,
  title={BELT: Old-School Backdoor Attacks can Evade the State-of-the-Art Defense with Backdoor Exclusivity Lifting},
  author={Qiu, Huming and Sun, Junjie and Zhang, Mi and Pan, Xudong and Yang, Min},
  journal={arXiv preprint arXiv:2312.04902},
  year={2023}
}

'''

import os
import sys
import yaml
from typing import *

sys.path = ["./"] + sys.path

import argparse
import numpy as np
import torch
import torch.nn as nn
import logging

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.useful_tools import register_hook_before_final

from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm

import copy

def add_common_attack_args(parser):
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    return parser


def get_mask_and_trigger(args):
    trans = transforms.Compose([
        transforms.Resize(args.img_size[:2]),  # (32, 32)
        np.array,
    ])

    trigger = trans(Image.open(args.patch_mask_path))
    mask = (trigger > 0)
    return mask, trigger



class Poison_Cover_Mix_Dataset(prepro_cls_DatasetBD_v2):
    def __init__(
            self,
            args,
            full_dataset_without_transform,
            poison_indicator: Optional[Sequence] = None,  # one-hot to determine which image may take bd_transform

            bd_image_pre_transform: Optional[Callable] = None,
            bd_label_pre_transform: Optional[Callable] = None,
            save_folder_path = None,

            mode = 'attack',
        ):

        super().__init__(
            deepcopy(full_dataset_without_transform),
            poison_indicator=None,
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            save_folder_path=save_folder_path,
        )

        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.target = args.attack_target

        self.bd_image_pre_transform = bd_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform

        self.pr = args.pratio
        self.cr = args.cover_rate
        self.mr = args.mask_rate

        n_pos = int(np.sum(poison_indicator))
        n_cov = int(n_pos * self.cr)
        logging.info(f"real number of poisoned image is {n_pos}, including {n_cov} cover images")

        pos_loc = (poison_indicator > 0)
        poison_indicator[pos_loc] = 1
        if n_cov > 0:
            pos_index = self.original_index_array[pos_loc]
            cov_index = np.random.choice(pos_index, n_cov, replace=False)
            poison_indicator[cov_index] = 2
        self.poison_indicator = poison_indicator

        self.mask, self.pattern = get_mask_and_trigger(args)

        self.pre_cover_transform = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])
        self.prepro_backdoor()

        self.getitem_all = True
        self.getitem_all_switch = False

        self.mode = mode



    def prepro_backdoor(self):
        for selected_index in tqdm(self.original_index_array, desc="prepro_backdoor"):
            if self.poison_indicator[selected_index] == 1:
                img, label, *additional_info = self.dataset[selected_index]
                img = self.bd_image_pre_transform(img, target=label, image_serial_id=selected_index)
                bd_label = self.bd_label_pre_transform(label)
                self.set_one_bd_sample(
                    selected_index, img, bd_label, label
                )
            elif self.poison_indicator[selected_index] == 2:
                img, label, *additional_info = self.dataset[selected_index]
                img = self.pre_cover_transform(img)

                mask = self.mask_mask(self.mr)
                img = img * (1 - mask) + self.pattern * mask
                img = np.clip(img, 0, 255)
                bd_label = label
                self.set_one_bd_sample(
                    selected_index, img, bd_label, label
                )


    def mask_mask(self, mask_rate):
        mask_flatten = copy.deepcopy(self.mask)[..., 0:1].reshape(-1)
        maks_temp = mask_flatten[mask_flatten != 0]
        maks_mask = np.random.permutation(maks_temp.shape[0])[:int(maks_temp.shape[0] * mask_rate)]
        maks_temp[maks_mask] = 0
        mask_flatten[mask_flatten != 0] = maks_temp
        mask_flatten = mask_flatten.reshape(self.mask[..., 0:1].shape)
        mask = np.repeat(mask_flatten, 3, axis=-1)
        return mask




class CenterLoss(nn.Module):
    def __init__(self, momentum=0.99, num_classes=10, device='cuda'):
        super(CenterLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.center = None
        self.radius = None
        self.momentum = momentum
        self.num_classes = num_classes
        self.device=device

    def update(self, features, targets, pmarks):
        if self.center is None:
            self.center = torch.zeros(self.num_classes, features.size(-1)).to(self.device)
            self.radius = torch.zeros(self.num_classes).to(self.device)

        features = features[pmarks == 0]
        targets = targets[pmarks == 0]

        for i in range(self.num_classes):
            features_i = features[targets == i]
            if features_i.size(0) > 0:
                self.center[i] = self.center[i] * self.momentum + features_i.mean(dim=0).detach() * (1 - self.momentum)
                # radius_i = torch.pairwise_distance(features_i, self.center[i], p=2)
                # self.radius[i] = self.radius[i] * self.momentum + radius_i.mean(dim=0).detach() * (1 - self.momentum)

    def forward(self, features, targets, pmarks):
        self.update(features, targets, pmarks)

        p_features = features[pmarks != 0]
        p_targets = targets[pmarks != 0]
        if p_features.size(0) != 0:
            loss = self.mse(p_features, self.center[p_targets].detach()).mean()
        else:
            loss = torch.zeros(1).to(self.device)
        return loss


class CenterLossTrainer(BackdoorModelTrainer):
    def __init__(self, model, args, center_loss_func=None):
        super().__init__(model)
        logging.debug("This class REQUIRE bd dataset to implement overwrite methods. This is NOT a general class for all cls task.")

        self.args = args
        if center_loss_func is None:
            self.center_loss_func = CenterLoss(momentum=args.center_loss_momentum, num_classes=args.num_classes, device=args.device)
        else:
            self.center_loss_func = center_loss_func

        self.record_array = list()
        self.final_module, self.hook_handle = None, None

        if args.center_loss_weight > 0:
            self.register_hooker()


    def register_hooker(self):
        final_module, hook_handle = register_hook_before_final(self.model, self.record_array, store_cpu=False, detach=False, clone=False)
        self.final_module = final_module
        self.hook_handle = hook_handle


    def one_forward_backward(self, x, labels, device, verbose=0, poison_indicator=None):

        self.model.train()
        self.model.to(device, non_blocking=self.non_blocking)

        x, labels = x.to(device, non_blocking=self.non_blocking), labels.to(device, non_blocking=self.non_blocking)

        if self.args.center_loss_weight > 0:
            self.record_array.clear()
        logits = self.model(x)
        loss = self.criterion(logits, labels.long())
        if self.args.center_loss_weight > 0:
            features = self.record_array[-1]
            center_loss = self.center_loss_func(features, labels, poison_indicator)
            loss += center_loss * self.args.center_loss_weight

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.args.center_loss_weight > 0:
            self.record_array.clear()

        batch_loss = loss.item()

        if verbose == 1:
            batch_predict = torch.max(logits, -1)[1].detach().clone().cpu()
            return batch_loss, batch_predict

        return batch_loss, None

    def test_given_dataloader_on_mix(self, test_dataloader, device = None, verbose = 0):
        has_hooker = False
        if self.hook_handle:
            self.hook_handle.remove()
            has_hooker = True
        rst = super().test_given_dataloader_on_mix(test_dataloader, device, verbose)
        if has_hooker:
            self.register_hooker()
        return rst



class Belt(NormalCase):
    def __init__(self):
        super(Belt).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)

        parser.add_argument("--patch_mask_path", type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/badnet/default.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument("--cover_rate", type=float) # default = 0.5
        parser.add_argument("--mask_rate", type=float) # default = 0.1
        parser.add_argument("--center_loss_weight", type=float) # default = 1.0
        return parser

    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, 'r') as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = mix_defaults

    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)

        ### 4. set the backdoor attack data and backdoor test data
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
        )

        logging.debug(f"poison train idx is saved")
        torch.save(train_poison_index,
                   args.save_path + '/train_poison_index_list.pickle',
                   )

        ### generate train dataset for backdoor attack
        bd_train_dataset = Poison_Cover_Mix_Dataset(
            args,
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        ### decide which img to poison in ASR Test
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        from torch.utils.data.dataloader import DataLoader
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='attack',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )


if __name__ == '__main__':
    attack = Belt()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
