# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

class HPUDeviceUtils:
    """HPU helpers."""

    @staticmethod
    # Gaudi HW performs convolution operations with filter (weights) in filters last format - RSCK format where:
    # R = height of the filter ,S = width of the filter, C = number of channels per filter,K = number of filters
    # The default PyTorch convolution weight ordering is ‘filters first’ (KCRS).
    # Therefore a re-ordering/permutation of all the convolution weights from KCRS to RSCK format is required
    # before convolution operations
    def permute_params(model, to_filters_last):
        """ permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
            and permute from RSCK to KCRS is used for checkpoint saving"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if(param.ndim == 4):
                    if to_filters_last:
                        param.data = param.data.permute((2, 3, 1, 0))
                    else:
                        param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS


    @staticmethod
    # permute the momentum from filters first (KCRS) to filters last(RSCK) or vice versa.
    # and permute from RSCK to KCRS is used for checkpoint saving
    def permute_momentum(optimizer, to_filters_last):
        # Permute the momentum buffer before using for checkpoint
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = optimizer.state[p]
                if 'momentum_buffer' in param_state:
                    buf = param_state['momentum_buffer']
                    if(buf.ndim == 4):
                        if to_filters_last:
                            buf = buf.permute((2,3,1,0))
                        else:
                            buf = buf.permute((3,2,0,1))
                        param_state['momentum_buffer'] = buf