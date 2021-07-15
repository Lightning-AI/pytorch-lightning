import os.path
import re
import tempfile

import yaml

from pytorch_lightning.loggers.dagshub import DAGsHubLogger


def in_tmp_dir(f):
    with tempfile.TemporaryDirectory() as d:
        metrics_path = os.path.join(d, 'metrics.csv')
        hparams_path = os.path.join(d, 'hparams.yml')
        f(metrics_path, hparams_path)


def test_context_manager_no_eager_logging():

    def f(metrics_path, hparams_path):
        with DAGsHubLogger(metrics_path=metrics_path, hparams_path=hparams_path, eager_logging=False) as logger:
            logger.log_metrics({'a': 1, 'b': 2})
            logger.log_hyperparams({'R': 2, 'D': 2})
            logger.log_metrics(a=3, c=42, step_num=2)
            logger.log_hyperparams(R=5, lr=1e-4)
            assert not os.path.exists(metrics_path)
            assert not os.path.exists(hparams_path)

        assert os.path.exists(metrics_path)
        assert os.path.exists(hparams_path)

        with open(metrics_path) as metrics_file:
            assert metrics_file.readline() == "Name,Value,Timestamp,Step\n"
            assert re.compile('^"a",1,\\d+,1\n$').match(metrics_file.readline())
            assert re.compile('^"b",2,\\d+,1\n$').match(metrics_file.readline())
            assert re.compile('^"a",3,\\d+,2\n$').match(metrics_file.readline())
            assert re.compile('^"c",42,\\d+,2\n$').match(metrics_file.readline())
            assert not metrics_file.readline()

        with open(hparams_path) as hparams_file:
            assert yaml.safe_load(hparams_file) == {'R': 5, 'D': 2, 'lr': 1e-4}

    in_tmp_dir(f)


def test_eager_logging():

    def f(metrics_path, hparams_path):
        logger = DAGsHubLogger(metrics_path=metrics_path, hparams_path=hparams_path, eager_logging=True)
        assert os.path.exists(metrics_path)
        assert os.path.exists(hparams_path)
        logger.close()

    in_tmp_dir(f)


def test_forbidden_csv_chars_in_metric_names():

    def f(metrics_path, hparams_path):
        with DAGsHubLogger(metrics_path=metrics_path, hparams_path=hparams_path) as logger:
            logger.log_metrics({
                'this/is/forbidden': 1,
                'so,is,this': 2,
                'and "this"': 3,
                'also \n this': 4,
                'normal': 5
            })

        lines = list(open(metrics_path))
        assert lines[0] == "Name,Value,Timestamp,Step\n"
        assert lines[1].startswith('"this/is/forbidden",1'), lines[1]
        assert lines[2].startswith('"so,is,this",2'), lines[2]
        assert lines[3].startswith('"and ""this""",3'), lines[3]
        assert lines[4] == '"also \n'
        assert lines[5].startswith(' this",4'), lines[5]
        assert lines[6].startswith('"normal",5'), lines[6]

    in_tmp_dir(f)
