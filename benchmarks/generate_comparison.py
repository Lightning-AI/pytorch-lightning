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

from benchmarks.test_basic_parity import measure_loops
from tests.base.models import ParityModuleMNIST, ParityModuleRNN

NUM_EPOCHS = 20
NUM_RUNS = 50
MODEL_CLASSES = (ParityModuleRNN, ParityModuleMNIST)
PATH_HERE = os.path.dirname(__file__)
FIGURE_EXTENSION = '.png'


def _main():
    fig, axarr = plt.subplots(nrows=len(MODEL_CLASSES))

    for i, cls_model in enumerate(MODEL_CLASSES):
        path_csv = os.path.join(PATH_HERE, f'dump-times_{cls_model.__name__}.csv')
        if os.path.isfile(path_csv):
            df_time = pd.read_csv(path_csv, index_col=0)
        else:
            # todo: kind="Vanilla PT" -> use_lightning=False
            vanilla = measure_loops(cls_model, kind="Vanilla PT", num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS)
            lightning = measure_loops(cls_model, kind="PT Lightning", num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS)

            df_time = pd.DataFrame({'vanilla PT': vanilla['durations'][1:], 'PT Lightning': lightning['durations'][1:]})
            df_time /= NUM_RUNS
            df_time.to_csv(os.path.join(PATH_HERE, f'dump-times_{cls_model.__name__}.csv'))
        # todo: add also relative X-axis ticks to see both: relative and absolute time differences
        df_time.plot.hist(
            ax=axarr[i],
            bins=20,
            alpha=0.5,
            title=cls_model.__name__,
            legend=True,
            grid=True,
        )
        axarr[i].set(xlabel='time [seconds]')

    path_fig = os.path.join(PATH_HERE, f'figure-parity-times{FIGURE_EXTENSION}')
    fig.tight_layout()
    fig.savefig(path_fig)


if __name__ == '__main__':
    _main()
