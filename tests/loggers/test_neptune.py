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
import pickle
import unittest
from collections import namedtuple
from unittest.mock import call, MagicMock, patch

import pytest
import torch

from pytorch_lightning import __version__, Trainer
from pytorch_lightning.loggers import NeptuneLogger
from tests.helpers import BoringModel


def create_neptune_mock():
    """Mock with provides nice `logger.name` and `logger.version` values.

    Mostly due to fact, that windows tests were failing with MagicMock based strings, which were used to create local
    directories in FS.
    """
    return MagicMock(
        init=MagicMock(
            return_value=MagicMock(
                __getitem__=MagicMock(return_value=MagicMock(fetch=MagicMock(return_value="Run test name"))),
                _short_id="TEST-1",
            )
        )
    )


class Run:
    _short_id = "TEST-42"
    _project_name = "test-project"

    def __setitem__(self, key, value):
        # called once
        assert key == "source_code/integrations/pytorch-lightning"
        assert value == __version__

    def wait(self):
        # for test purposes
        pass

    def __getitem__(self, item):
        # called once
        assert item == "sys/name"
        return MagicMock(fetch=MagicMock(return_value="Test name"))

    def __getstate__(self):
        raise pickle.PicklingError("Runs are unpickleable")


@pytest.fixture
def tmpdir_unittest_fixture(request, tmpdir):
    """Proxy for pytest `tmpdir` fixture between pytest and unittest.
    Resources:
     * https://docs.pytest.org/en/6.2.x/tmpdir.html#the-tmpdir-fixture
     * https://towardsdatascience.com/mixing-pytest-fixture-and-unittest-testcase-for-selenium-test-9162218e8c8e
    """
    request.cls.tmpdir = tmpdir


@patch("pytorch_lightning.loggers.neptune.neptune", new_callable=create_neptune_mock)
class TestNeptuneLogger(unittest.TestCase):
    def test_neptune_online(self, neptune):
        logger = NeptuneLogger(api_key="test", project="project")
        created_run_mock = logger._run_instance

        self.assertEqual(logger._run_instance, created_run_mock)
        self.assertEqual(logger.name, "Run test name")
        self.assertEqual(logger.version, "TEST-1")
        self.assertEqual(neptune.init.call_count, 1)
        self.assertEqual(created_run_mock.__getitem__.call_count, 1)
        self.assertEqual(created_run_mock.__setitem__.call_count, 1)
        created_run_mock.__getitem__.assert_called_once_with(
            "sys/name",
        )
        created_run_mock.__setitem__.assert_called_once_with("source_code/integrations/pytorch-lightning", __version__)

    @patch("pytorch_lightning.loggers.neptune.Run", Run)
    def test_online_with_custom_run(self, neptune):
        created_run = Run()
        logger = NeptuneLogger(run=created_run)

        assert logger._run_instance == created_run
        self.assertEqual(logger._run_instance, created_run)
        self.assertEqual(logger.version, created_run._short_id)
        self.assertEqual(neptune.init.call_count, 0)

    @patch("pytorch_lightning.loggers.neptune.Run", Run)
    def test_neptune_pickling(self, neptune):
        unpickleable_run = Run()
        logger = NeptuneLogger(run=unpickleable_run)
        self.assertEqual(0, neptune.init.call_count)

        pickled_logger = pickle.dumps(logger)
        unpickled = pickle.loads(pickled_logger)

        neptune.init.assert_called_once_with(project="test-project", api_token=None, run="TEST-42")
        self.assertIsNotNone(unpickled.experiment)

    @patch("pytorch_lightning.loggers.neptune.Run", Run)
    def test_online_with_wrong_kwargs(self, neptune):
        """Tests combinations of kwargs together with `run` kwarg which makes some of other parameters unavailable
        in init."""
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
        trainer = Trainer(default_root_dir=self.tmpdir, max_epochs=1, limit_train_batches=0.05, logger=logger)
        assert trainer.log_dir == os.path.join(os.getcwd(), ".neptune")
        trainer.fit(model)
        trainer.test(model)
        assert trainer.log_dir == os.path.join(os.getcwd(), ".neptune")

    @pytest.mark.usefixtures("tmpdir_unittest_fixture")
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

    @pytest.mark.usefixtures("tmpdir_unittest_fixture")
    def test_neptune_log_metrics_on_trained_model(self, neptune):
        """Verify that trained models do log data."""
        # given
        class LoggingModel(BoringModel):
            def validation_epoch_end(self, outputs):
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
        run_instance_mock.__getitem__.return_value.log.assert_has_calls([call(42)])

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
            self.assertEqual(run_instance_mock.__setitem__.call_count, 1)
            self.assertEqual(run_instance_mock.__getitem__.call_count, 0)
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
            self.assertEqual(run_instance_mock.__setitem__.call_count, 0)
            self.assertEqual(run_instance_mock.__getitem__.call_count, 2)
            run_instance_mock.__getitem__.assert_any_call(metrics_foo_key)
            run_instance_mock.__getitem__.assert_any_call(metrics_bar_key)
            run_attr_mock.log.assert_has_calls([call(42), call(555)])

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
            self.assertEqual(run_instance_mock.__setitem__.call_count, 1)
            self.assertEqual(run_instance_mock.__getitem__.call_count, 0)
            run_instance_mock.__setitem__.assert_called_once_with(model_summary_key, file_from_content_mock)

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
            cb_mock = MagicMock(
                dirpath="path/to/models",
                last_model_path="path/to/models/last",
                best_k_models={
                    "path/to/models/model1": None,
                    "path/to/models/model2/with/slashes": None,
                },
                best_model_path="path/to/models/best_model",
                best_model_score=None,
            )

            # when: save checkpoint
            logger.after_save_checkpoint(cb_mock)

            # then:
            self.assertEqual(run_instance_mock.__setitem__.call_count, 1)
            self.assertEqual(run_instance_mock.__getitem__.call_count, 3)
            self.assertEqual(run_attr_mock.upload.call_count, 3)
            run_instance_mock.__setitem__.assert_called_once_with(
                f"{model_key_prefix}/best_model_path", "path/to/models/best_model"
            )
            run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/last")
            run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/model1")
            run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/model2/with/slashes")
            run_attr_mock.upload.assert_has_calls(
                [
                    call("path/to/models/last"),
                    call("path/to/models/model1"),
                    call("path/to/models/model2/with/slashes"),
                ]
            )

    def test_save_dir(self, neptune):
        # given
        logger = NeptuneLogger(api_key="test", project="project")

        # expect
        self.assertEqual(logger.save_dir, os.path.join(os.getcwd(), ".neptune"))


class TestNeptuneLoggerDeprecatedUsages(unittest.TestCase):
    @staticmethod
    def _assert_legacy_usage(callback, *args, **kwargs):
        with pytest.raises(ValueError):
            callback(*args, **kwargs)

    def test_legacy_kwargs(self):
        legacy_neptune_kwargs = [
            # NeptuneLegacyLogger kwargs
            "project_name",
            "offline_mode",
            "experiment_name",
            "experiment_id",
            "params",
            "properties",
            "upload_source_files",
            "abort_callback",
            "logger",
            "upload_stdout",
            "upload_stderr",
            "send_hardware_metrics",
            "run_monitoring_thread",
            "handle_uncaught_exceptions",
            "git_info",
            "hostname",
            "notebook_id",
            "notebook_path",
            # NeptuneLogger from neptune-pytorch-lightning package kwargs
            "base_namespace",
            "close_after_fit",
        ]
        for legacy_kwarg in legacy_neptune_kwargs:
            self._assert_legacy_usage(NeptuneLogger, **{legacy_kwarg: None})

    @patch("pytorch_lightning.loggers.neptune.warnings")
    @patch("pytorch_lightning.loggers.neptune.NeptuneFile")
    @patch("pytorch_lightning.loggers.neptune.neptune")
    def test_legacy_functions(self, neptune, neptune_file_mock, warnings_mock):
        logger = NeptuneLogger(api_key="test", project="project")

        # test deprecated functions which will be shut down in pytorch-lightning 1.7.0
        attr_mock = logger._run_instance.__getitem__
        attr_mock.reset_mock()
        fake_image = {}

        logger.log_metric("metric", 42)
        logger.log_text("text", "some string")
        logger.log_image("image_obj", fake_image)
        logger.log_image("image_str", "img/path")
        logger.log_artifact("artifact", "some/path")

        assert attr_mock.call_count == 5
        assert warnings_mock.warn.call_count == 5
        attr_mock.assert_has_calls(
            [
                call("training/metric"),
                call().log(42, step=None),
                call("training/text"),
                call().log("some string", step=None),
                call("training/image_obj"),
                call().log(fake_image, step=None),
                call("training/image_str"),
                call().log(neptune_file_mock(), step=None),
                call("training/artifacts/artifact"),
                call().log("some/path"),
            ]
        )

        # test Exception raising functions  functions
        self._assert_legacy_usage(logger.set_property)
        self._assert_legacy_usage(logger.append_tags)


class TestNeptuneLoggerUtils(unittest.TestCase):
    def test__get_full_model_name(self):
        # given:
        SimpleCheckpoint = namedtuple("SimpleCheckpoint", ["dirpath"])
        test_input_data = [
            ("key.ext", "foo/bar/key.ext", SimpleCheckpoint(dirpath="foo/bar")),
            ("key/in/parts.ext", "foo/bar/key/in/parts.ext", SimpleCheckpoint(dirpath="foo/bar")),
        ]

        # expect:
        for expected_model_name, *key_and_path in test_input_data:
            self.assertEqual(NeptuneLogger._get_full_model_name(*key_and_path), expected_model_name)

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
        self.assertEqual(NeptuneLogger._get_full_model_names_from_exp_structure(input_dict, "foo/bar"), expected_keys)
