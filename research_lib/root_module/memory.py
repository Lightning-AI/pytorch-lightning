import torch
import gc
import subprocess
import numpy as np
import pandas as pd


'''
Generates a summary of a model's layers and dimensionality
'''


class ModelSummary(object):

    def __init__(self, model):
        '''
        Generates summaries of model layers and dimensions.
        '''
        self.model = model
        self.in_sizes = []
        self.out_sizes = []

        self.summarize()

    def __str__(self):
        return self.summary.__str__()

    def __repr__(self):
        return self.summary.__str__()

    def get_variable_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        mods = list(self.model.modules())
        in_sizes = []
        out_sizes = []
        input_ = self.example_input_array
        for i in range(1, len(mods)):
            m = mods[i]
            if type(input_) is list or type(input_) is tuple:
                out = m(*input_)
            else:
                out = m(input_)

            if type(input_) is tuple or type(input_) is list:
                in_size = []
                for x in input_:
                    if type(x) is list:
                        in_size.append(len(x))
                    else:
                        in_size.append(x.size())
            else:
                in_size = np.array(input_.size())

            in_sizes.append(in_size)

            if type(out) is tuple or type(out) is list:
                out_size = np.asarray([x.size() for x in out])
            else:
                out_size = np.array(out.size())

            out_sizes.append(out_size)
            input_ = out

        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        return

    def get_layer_names(self):
        '''Collect Layer Names'''
        mods = list(self.model.named_modules())
        names = []
        layers = []
        for m in mods[1:]:
            names += [m[0]]
            layers += [str(m[1].__class__)]

        layer_types = [x.split('.')[-1][:-2] for x in layers]

        self.layer_names = names
        self.layer_types = layer_types
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            modsz = []
            for j in range(len(p)):
                modsz.append(np.array(p[j].size()))
            sizes.append(modsz)

        self.param_sizes = sizes
        return

    def get_parameter_nums(self):
        '''Get number of parameters in each layer'''
        param_nums = []
        for mod in self.param_sizes:
            all_params = 0
            for p in mod:
                all_params += np.prod(p)
            param_nums.append(all_params)
        self.param_nums = param_nums
        return

    def make_summary(self):
        '''
        Makes a summary listing with:

        Layer Name, Layer Type, Input Size, Output Size, Number of Parameters
        '''

        df = pd.DataFrame( np.zeros( (len(self.layer_names), 3) ) )
        df.columns = ['Name', 'Type', 'Params']

        df['Name'] = self.layer_names
        df['Type'] = self.layer_types
        df['Params'] = self.param_nums

        self.summary = df
        return

    def summarize(self):
        self.get_layer_names()
        self.get_parameter_sizes()
        self.get_parameter_nums()
        self.make_summary()


def print_mem_stack():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except Exception as e:
            pass


def count_mem_items():
    nb_params = 0
    nb_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_type = str(type(obj))
                if 'parameter' in obj_type:
                    nb_params += 1
                else:
                    nb_tensors += 1
        except Exception as e:
            pass

    return nb_params, nb_tensors


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = {}
    for k, v in zip(range(len(gpu_memory)), gpu_memory):
        k = f'gpu_{k}'
        gpu_memory_map[k] = v
    return gpu_memory_map
