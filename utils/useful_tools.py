import os
import torch

def absolute_to_relative_path(abs_path, after='BackdoorBench'):
    p = os.path.normpath(abs_path)
    ds = p.split(os.sep)
    kk = None
    for k, d in enumerate(ds):
        if d==after:
            kk = k
    if kk and kk < len(ds)-1:
        new_p = os.sep.join(ds[kk+1:])
        return new_p
    else:
        return abs_path

def net_to_device(args, net):
    if hasattr(args, 'doDataParallel') and args.doDataParallel:
        net = torch.nn.DataParallel(net)
    net.to(args.device)
    return net

def build_record_hooker(record_array, store_cpu=False, detach=False, clone=False):
    def hooker(module, input):
        z = input[0]
        if detach:
            z = z.detach()
        if store_cpu:
            z = z.cpu()
        if clone:
            z = z.clone()
        record_array.append(z)
    return hooker

def register_hook_before_final(clsmodel, record_array, store_cpu=False, detach=False, clone=False):
    m_list = list(clsmodel.modules())
    m_list.reverse()
    for m in m_list:
        if isinstance(m, torch.nn.Linear):
            final_module = m
            break
    hooker = build_record_hooker(record_array, store_cpu, detach, clone)
    hook_handle = final_module.register_forward_pre_hook(hooker)
    return final_module, hook_handle


class RandomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 data_shape,
                 len,
                 random_type,
                 num_classes=None,
                 fully_random=False
         ):
        self.data_shape = data_shape
        self.n = len

        if random_type == 'uniform':
            self.rand_gen = torch.rand
        elif random_type == 'normal':
            self.rand_gen = torch.randn
        else:
            raise NotImplementedError
        
        self.fully_random = fully_random
        
        if not fully_random:
            self.data = dict()

        self.num_classes = num_classes

    def __getitem__(self, index):
        if self.fully_random:
            data = self.rand_gen(self.data_shape)
        else:
            if index not in self.data:
                self.data[index] = self.rand_gen(self.data_shape)
            data = self.data[index]

        if self.num_classes is not None:
            target = torch.randint(self.num_classes, (1,)).item()
            return data, target
        else:
            return data

    def __len__(self):
        return self.n

