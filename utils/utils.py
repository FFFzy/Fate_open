import gc
import torch
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.untyped_storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.untyped_storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


def torch_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        print(tensor.shape, tensor.data_ptr())

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem
