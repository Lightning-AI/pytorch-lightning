# Copyright The Lightning AI team.
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
import pickle
import unittest
from collections import namedtuple
from unittest import mock
from unittest.mock import call, MagicMock, patch

import pytest
import torch

from lightning.pytorch import __version__, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import NeptuneLogger


def fetchable_paths(value):
    if value == "sys/id":
        return MagicMock(fetch=MagicMock(return_value="TEST-1"))
    if value == "sys/name":
        return MagicMock(fetch=MagicMock(return_value="Run test name"))
    return MagicMock()


def create_run_mock(mode="async", **kwargs):
    if mode == "offline":
        return MagicMock(__getitem__=MagicMock(side_effect=fetchable_paths), exists=MagicMock(return_value=False))
    return MagicMock(__getitem__=MagicMock(side_effect=fetchable_paths), exists=MagicMock(return_value=True))


def create_neptune_mock():
    """Mock with provides nice `logger.name` and `logger.version` values. Additionally, it allows `mode` as an argument
    to test different Neptune modes.

    Mostly due to fact, that windows tests were failing with MagicMock based strings, which were used to create local
    directories in FS.

    """
    return MagicMock(init_run=MagicMock(side_effect=create_run_mock))


# Note: For testing purpose this mock `Run` will also be used to mock `Handler`.
class Run:
    _project_name = "test-project"

    def __setitem__(self, key, value):
        # called once
        assert key == "source_code/integrations/pytorch-lightning"
        assert value == __version__

    def wait(self):
        # for test purposes
        pass

    def __getitem__(self, item):
        if item == "sys/name":
            return MagicMock(fetch=MagicMock(return_value="Test name"))
        if item == "sys/id":
            return MagicMock(fetch=MagicMock(return_value="TEST-42"))

        pytest.fail(f"Unexpected call '{item}'")

    def __getstate__(self):
        raise pickle.PicklingError("Runs are unpickleable")

    def exists(self, value):
        return True

    def get_root_object(self):
        return self


@pytest.fixture()
def tmp_path_unittest_fixture(request, tmp_path):
    """Proxy for pytest `tmp_path` fixture between pytest and unittest.

    Resources:
     * https://docs.pytest.org/en/6.2.x/tmp_path.html#the-tmp_path-fixture
     * https://towardsdatascience.com/mixing-pytest-fixture-and-unittest-testcase-for-selenium-test-9162218e8c8e

    """
    request.cls.tmp_path = tmp_path


@patch("lightning.pytorch.loggers.neptune.neptune", new_callable=create_neptune_mock)
class TestNeptuneLogger(unittest.TestCase):
    def run(self, *args, **kwargs):
        with mock.patch("lightning.pytorch.loggers.neptune._NEPTUNE_AVAILABLE", return_value=True):
            super().run(*args, **kwargs)

    @patch("lightning.pytorch.loggers.neptune.Run", Run)
    @patch("lightning.pytorch.loggers.neptune.Handler", Run)
    def test_neptune_online(self, neptune):
        logger = NeptuneLogger(api_key="test", project="project")
        created_run_mock = logger.run

        assert logger._run_instance == created_run_mock
        created_run_mock.exists.assert_called_once_with("sys/id")
        assert logger.name == "Run test name"
        assert logger.version == "TEST-1"
        assert neptune.init_run.call_count == 1
        assert created_run_mock.__getitem__.call_count == 2
        assert created_run_mock.__setitem__.call_count == 1
        created_run_mock.__getitem__.assert_has_calls([call("sys/id"), call("sys/name")], any_order=True)
        created_run_mock.__setitem__.assert_called_once_with("source_code/integrations/pytorch-lightning", __version__)

    @patch("lightning.pytorch.loggers.neptune.Run", Run)
    @patch("lightning.pytorch.loggers.neptune.Handler", Run)
    def test_neptune_offline(self, neptune):
        logger = NeptuneLogger(mode="offline")
        created_run_mock = logger.run
        logger.experiment["foo"] = "bar"

        created_run_mock.exists.assert_called_once_with("sys/id")
        assert logger._run_short_id == "OFFLINE"
        assert logger._run_name == "offline-name"

    @patch("lightning.pytorch.loggers.neptune.Run", Run)
    @patch("lightning.pytorch.loggers.neptune.Handler", Run)
    def test_online_with_custom_run(self, neptune):
        created_run = Run()
        logger = NeptuneLogger(run=created_run)

        assert logger._run_instance == created_run
        assert logger._run_instance == created_run
        assert logger.version == "TEST-42"
        assert neptune.init_run.call_count == 0

    @patch("lightning.pytorch.loggers.neptune.Run", Run)
    @patch("lightning.pytorch.loggers.neptune.Handler", Run)
    def test_neptune_pickling(self, neptune):
        unpickleable_run = Run()
        logger = NeptuneLogger(run=unpickleable_run)
        assert neptune.init_run.call_count == 0

        pickled_logger = pickle.dumps(logger)
        unpickled = pickle.loads(pickled_logger)

        neptune.init_run.assert_called_once_with(name="Test name", run="TEST-42")
        assert unpickled.experiment is not None

    @patch("lightning.pytorch.loggers.neptune.Run", Run)
    @patch("lightning.pytorch.loggers.neptune.Handler", Run)
    def test_online_with_wrong_kwargs(self, neptune):
        """Tests combinations of kwargs together with `run` kwarg which makes some of other parameters unavailable in
        init."""
        with self.assertRaises(ValueError):
            NeptuneLogger(run="some string")

        with self.assertRaises(ValueError):
            NeptuneLogger(run=Run(), project="redundant project")

        with self.assertRaises(ValueError):
            NeptuneLogger(run=Run(), api_key="redundant api key")

        with self.assertRaises(ValueError):
            NeptuneLogger(run=Run(), name="redundant api name")

        with self.assertRaises(ValueError):
            NeptuneLogger(run=Run(), foo="random **kwarg")

        # this should work
        NeptuneLogger(run=Run())
        NeptuneLogger(project="foo")
        NeptuneLogger(foo="bar")

    @staticmethod
    def _get_logger_with_mocks(**kwargs):
        logger = NeptuneLogger(**kwargs)
        run_instance_mock = MagicMock()
        logger._run_instance = run_instance_mock
        logger._run_instance.__getitem__.return_value.fetch.return_value = "exp-name"
        run_attr_mock = MagicMock()
        logger._run_instance.__getitem__.return_value = run_attr_mock

        return logger, run_instance_mock, run_attr_mock

    def test_neptune_additional_methods(self, neptune):
        logger, run_instance_mock, _ = self._get_logger_with_mocks(api_key="test", project="project")

        logger.experiment["key1"].log(torch.ones(1))
        run_instance_mock.__getitem__.assert_called_once_with("key1")
        run_instance_mock.__getitem__().log.assert_called_once_with(torch.ones(1))

    def _fit_and_test(self, logger, model):
        trainer = Trainer(default_root_dir=self.tmp_path, max_epochs=1, limit_train_batches=0.05, logger=logger)
        assert trainer.log_dir == os.path.join(os.getcwd(), ".neptune")
        trainer.fit(model)
        trainer.test(model)
        assert trainer.log_dir == os.path.join(os.getcwd(), ".neptune")

    @pytest.mark.usefixtures("tmp_path_unittest_fixture")
    @patch("lightning.pytorch.loggers.neptune.File", new=mock.Mock())
    def test_neptune_leave_open_experiment_after_fit(self, neptune):
        """Verify that neptune experiment was NOT closed after training."""
        # given
        logger, run_instance_mock, _ = self._get_logger_with_mocks(api_key="test", project="project")

        # when
        self._fit_and_test(
            logger=logger,
            model=BoringModel(),
        )

        # then
        assert run_instance_mock.stop.call_count == 0

    @pytest.mark.usefixtures("tmp_path_unittest_fixture")
    @patch("lightning.pytorch.loggers.neptune.File", new=mock.Mock())
    def test_neptune_log_metrics_on_trained_model(self, neptune):
        """Verify that trained models do log data."""

        # given
        class LoggingModel(BoringModel):
            def on_validation_epoch_end(self):
                self.log("some/key", 42)

        # and
        logger, run_instance_mock, _ = self._get_logger_with_mocks(api_key="test", project="project")

        # when
        self._fit_and_test(
            logger=logger,
            model=LoggingModel(),
        )

        # then
        run_instance_mock.__getitem__.assert_any_call("training/some/key")
        run_instance_mock.__getitem__.return_value.append.assert_has_calls([call(42)])

    @patch("lightning.pytorch.loggers.neptune.stringify_unsupported", lambda x: x)
    def test_log_hyperparams(self, neptune):
        params = {"foo": "bar", "nested_foo": {"bar": 42}}
        test_variants = [
            ({}, "training/hyperparams"),
            ({"prefix": "custom_prefix"}, "custom_prefix/hyperparams"),
            ({"prefix": "custom/nested/prefix"}, "custom/nested/prefix/hyperparams"),
        ]
        for prefix, hyperparams_key in test_variants:
            # given:
            logger, run_instance_mock, _ = self._get_logger_with_mocks(api_key="test", project="project", **prefix)

            # when: log hyperparams
            logger.log_hyperparams(params)

            # then
            assert run_instance_mock.__setitem__.call_count == 1
            assert run_instance_mock.__getitem__.call_count == 0
            run_instance_mock.__setitem__.assert_called_once_with(hyperparams_key, params)

    def test_log_metrics(self, neptune):
        metrics = {
            "foo": 42,
            "bar": 555,
        }
        test_variants = [
            ({}, ("training/foo", "training/bar")),
            ({"prefix": "custom_prefix"}, ("custom_prefix/foo", "custom_prefix/bar")),
            ({"prefix": "custom/nested/prefix"}, ("custom/nested/prefix/foo", "custom/nested/prefix/bar")),
        ]

        for prefix, (metrics_foo_key, metrics_bar_key) in test_variants:
            # given:
            logger, run_instance_mock, run_attr_mock = self._get_logger_with_mocks(
                api_key="test", project="project", **prefix
            )

            # when: log metrics
            logger.log_metrics(metrics)

            # then:
            assert run_instance_mock.__setitem__.call_count == 0
            assert run_instance_mock.__getitem__.call_count == 2
            run_instance_mock.__getitem__.assert_any_call(metrics_foo_key)
            run_instance_mock.__getitem__.assert_any_call(metrics_bar_key)
            run_attr_mock.append.assert_has_calls([call(42), call(555)])

    def test_log_model_summary(self, neptune):
        model = BoringModel()
        test_variants = [
            ({}, "training/model/summary"),
            ({"prefix": "custom_prefix"}, "custom_prefix/model/summary"),
            ({"prefix": "custom/nested/prefix"}, "custom/nested/prefix/model/summary"),
        ]

        for prefix, model_summary_key in test_variants:
            # given:
            logger, run_instance_mock, _ = self._get_logger_with_mocks(api_key="test", project="project", **prefix)
            file_from_content_mock = neptune.types.File.from_content()

            # when: log metrics
            logger.log_model_summary(model)

            # then:
            assert run_instance_mock.__setitem__.call_count == 1
            assert run_instance_mock.__getitem__.call_count == 0
            run_instance_mock.__setitem__.assert_called_once_with(model_summary_key, file_from_content_mock)

    @patch("builtins.open", mock.mock_open(read_data="test"))
    def test_after_save_checkpoint(self, neptune):
        test_variants = [
            ({}, "training/model"),
            ({"prefix": "custom_prefix"}, "custom_prefix/model"),
            ({"prefix": "custom/nested/prefix"}, "custom/nested/prefix/model"),
        ]

        for prefix, model_key_prefix in test_variants:
            # given:
            logger, run_instance_mock, run_attr_mock = self._get_logger_with_mocks(
                api_key="test", project="project", **prefix
            )
            models_root_dir = os.path.join("path", "to", "models")
            cb_mock = MagicMock(
                dirpath=models_root_dir,
                last_model_path=os.path.join(models_root_dir, "last"),
                best_k_models={
                    f"{os.path.join(models_root_dir, 'model1')}": None,
                    f"{os.path.join(models_root_dir, 'model2/with/slashes')}": None,
                },
                best_model_path=os.path.join(models_root_dir, "best_model"),
                best_model_score=None,
            )

            with patch("lightning.pytorch.loggers.neptune.File", side_effect=mock.Mock()) as mock_file:
                # when: save checkpoint
                logger.after_save_checkpoint(cb_mock)

            # then:
            assert run_instance_mock.__setitem__.call_count == 3
            assert run_instance_mock.__getitem__.call_count == 2
            assert run_attr_mock.upload.call_count == 2

            assert mock_file.from_stream.call_count == 2

            run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/model1")
            run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/model2/with/slashes")

            run_attr_mock.upload.assert_has_calls(
                [
                    call(os.path.join(models_root_dir, "model1")),
                    call(os.path.join(models_root_dir, "model2/with/slashes")),
                ]
            )

    def test_save_dir(self, neptune):
        # given
        logger = NeptuneLogger(api_key="test", project="project")

        # expect
        assert logger.save_dir == os.path.join(os.getcwd(), ".neptune")


class TestNeptuneLoggerUtils(unittest.TestCase):
    def test__get_full_model_name(self):
        # given:
        SimpleCheckpoint = namedtuple("SimpleCheckpoint", ["dirpath"])
        test_input_data = [
            ("key", os.path.join("foo", "bar", "key.ext"), SimpleCheckpoint(dirpath=os.path.join("foo", "bar"))),
            (
                "key/in/parts",
                os.path.join("foo", "bar", "key/in/parts.ext"),
                SimpleCheckpoint(dirpath=os.path.join("foo", "bar")),
            ),
        ]

        # expect:
        for expected_model_name, *key_and_path in test_input_data:
            assert NeptuneLogger._get_full_model_name(*key_and_path) == expected_model_name

    def test__get_full_model_names_from_exp_structure(self):
        # given:
        input_dict = {
            "foo": {
                "bar": {
                    "lvl1_1": {"lvl2": {"lvl3_1": "some non important value", "lvl3_2": "some non important value"}},
                    "lvl1_2": "some non important value",
                },
                "other_non_important": {"val100": 100},
            },
            "other_non_important": {"val42": 42},
        }
        expected_keys = {"lvl1_1/lvl2/lvl3_1", "lvl1_1/lvl2/lvl3_2", "lvl1_2"}

        # expect:
        assert NeptuneLogger._get_full_model_names_from_exp_structure(input_dict, "foo/bar") == expected_keys
