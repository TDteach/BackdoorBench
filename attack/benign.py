'''
Benign: train model without backdoor

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. save the attack result for defense


'''

import os
import sys
import yaml

sys.path = ["./"] + sys.path

import argparse
import numpy as np
import torch
import logging

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import PureCleanModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from torch.utils.data.dataloader import DataLoader


class Benign(NormalCase):
    r"""Badnets: Identifying vulnerabilities in the machine learning model supply chain.

        basic structure:

        1. config args, save_path, fix random seed
        2. set the clean train data and clean test data
        3. set the attack img transform and label transform
        4. set the backdoor attack data and backdoor test data
        5. set the device, model, criterion, optimizer, training schedule.
        6. attack or use the model to do finetune with 5% clean data
        7. save the attack result for defense

        .. code-block:: python
            attack = BadNet()
            attack.attack()

        .. Note::
            @article{gu2017badnets,
            title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
            author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
            journal={arXiv preprint arXiv:1708.06733},
            year={2017}}

        Args:
            attack (string): name of attack, use to match the transform and set the saving prefix of path.
            attack_target (Int): target class No. in all2one attack
            attack_label_trans (str): which type of label modification in backdoor attack
            pratio (float): the poison rate
            bd_yaml_path (string): path for yaml file provide additional default attributes
            patch_mask_path (string): path for patch mask
            **kwargs (optional): Additional attributes.

        """

    def __init__(self):
        super(Benign).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/benign/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def add_bd_yaml_to_args(self, args):
        if not os.path.exists(args.bd_yaml_path):
            logging.info('no bd_yaml_path provided, use default settings instead')
            return
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

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              train_dataset_without_transform, \
                              test_dataset_without_transform

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        clean_train_dataset_without_transform, \
        clean_test_dataset_without_transform = self.stage1_results

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

        trainer = PureCleanModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
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
            bd_train=None,
            bd_test=None,
            save_path=args.save_path,
        )


if __name__ == '__main__':
    attack = Benign()
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
