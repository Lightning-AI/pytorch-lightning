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
import os

import matplotlib.pylab as plt
import pandas as pd

from benchmarks.test_basic_parity import vanilla_loop, lightning_loop
from tests.base.models import ParityModuleRNN, ParityModuleMNIST

NUM_EPOCHS = 20
NUM_RUNS = 50
MODEL_CLASSES = (ParityModuleRNN, ParityModuleMNIST)
PATH_HERE = os.path.dirname(__file__)


def _main():
    fig, axarr = plt.subplots(nrows=len(MODEL_CLASSES))

    for i, cls_model in enumerate(MODEL_CLASSES):
        path_csv = os.path.join(PATH_HERE, f'dump-times_{cls_model.__name__}.csv')
        if os.path.isfile(path_csv):
            df_time = pd.read_csv(path_csv)
        else:
            vanilla = vanilla_loop(cls_model, num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS)
            lightning = lightning_loop(cls_model, num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS)

            df_time = pd.DataFrame({'vanilla PT': vanilla['durations'], 'PT Lightning': lightning['durations']})
            df_time /= NUM_RUNS
            df_time.to_csv(os.path.join(PATH_HERE, f'dump-times_{cls_model.__name__}.csv'))
        # todo
        df_time.plot.hist(ax=axarr[i], bins=12, alpha=0.5)

    path_fig = os.path.join(PATH_HERE, 'figure-times.svg')
    fig.savefig(path_fig)

if __name__ == '__main__':
    _main()
