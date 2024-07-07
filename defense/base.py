import os,sys
import numpy as np
import torch
import logging
import time

import yaml

from pprint import pformat

from utils.aggregate_block.fix_random import fix_random
from utils.log_assist import get_git_info

from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape

from utils.useful_tools import net_to_device


class defense(object):

    def __init__(self,):
        # TODO:yaml config log(测试两个防御方法同时使用会不会冲突)
        print(1)

    def add_base_arguments(parser):
        parser.add_argument('--device', type=str, help='cpu or cuda', choices={"cpu","cuda"})
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--attack_folder', type=str, help='path to folder containing attack model')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/default/config.yaml", help='the path of yaml')

        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=int)

        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--log_level", dest="logLevel", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")
        parser.add_argument('--random_seed', type=int, help='random seed', default=0)
        parser.add_argument('--result_file', type=str, help='the location of result')
        return parser

    def add_arguments(parser):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以实现给参数的功能
        print('You need to rewrite this method for passing parameters')
    
    def set_result(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以读取攻击的结果
        print('You need to rewrite this method to load the attack result')
        
    def set_trainer(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现整合训练模块的功能
        print('If you want to use standard trainer module, please rewrite this method')
    
    def denoising(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def mitigation(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def inhibition(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def defense(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def detect(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')


    def set_logger(self, args):
        logLevel = args.logLevel
        logFormatter = logging.Formatter(
			fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
			datefmt='%Y-%m-%d:%H:%M:%S',
		)
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logLevel)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logLevel)
        logger.addHandler(consoleHandler)

        logger.setLevel(logLevel)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

        ### set the random seed
        fix_random(int(args.random_seed))

	
    def set_devices(self, args):
        # self.device = torch.device(
        # 	(
        # 		f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        # 		# since DataParallel only allow .to("cuda")
        # 	) if torch.cuda.is_available() else "cpu"
        # )
        self.device= self.args.device

    def set_paths_and_folders(self, args):
        result_file = args.attack_folder
        save_path = 'record/' + result_file + f'/defense/{self.__class__.__name__}/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        args.save_path = save_path
        if not hasattr(args, 'checkpoint_save') or args.checkpoint_save is None:
            args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(args.checkpoint_save)):
                os.makedirs(args.checkpoint_save) 
        if not hasattr(args, 'log') or args.log is None:
            args.log = save_path + 'log/'
            if not (os.path.exists(args.log)):
                os.makedirs(args.log)  
 

    def process_args(args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        if args.device=='cuda' and torch.cuda.device_count() > 1:
            setattr(args, 'doDataParallel', True)
        else:
            setattr(args, 'doDataParallel', False)
        return args
    
    def add_yaml_to_args(args):
        with open(args.yaml_path, 'r') as f:
            clean_defaults = yaml.safe_load(f)
        clean_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = clean_defaults
 

    def prepare(self, args=None):
        if args is None:
            args = self.args
        assert args is not None, "No args is provided for prepare()"

        self.set_paths_and_folders(args)
        self.set_devices(args)
        self.set_logger(args)
        fix_random(args.random_seed)

        self.args = args


