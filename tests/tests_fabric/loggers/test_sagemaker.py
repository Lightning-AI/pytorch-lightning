from typing import Any, List, Tuple, Union

import boto3
import numpy as np
import pytest
import torch
from moto import mock_sagemaker
from sagemaker.experiments.run import Run
from sagemaker.session import Session

from experiments_addon.logger import (
    SagemakerExperimentsLogger,
    _prep_param_for_serialization,
)

EXPERIMENT_NAME = "testexperiment"
RUN_NAME = "testrunname"





@pytest.fixture
def sagemaker_session():
    with mock_sagemaker():
        session = Session(boto3.Session(region_name="eu-central-1"))
        client = boto3.client("sagemaker", region_name="eu-central-1")
        yield session, client


@pytest.fixture
def sme_logger(
    sagemaker_session, mocker
) -> Tuple[SagemakerExperimentsLogger, Run]:
    mocker.patch("sagemaker.experiments.trial_component._TrialComponent.save")
    with Run(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        sagemaker_session=sagemaker_session[0],
    ) as run:
        yield SagemakerExperimentsLogger(
            sagemaker_session=sagemaker_session[0]
        ), run


@pytest.fixture
def binary_labels() -> Tuple[List, List]:
    y_true = [1, 0, 1, 0, 1]
    pred_proba = [0.8, 0.2, 0.2, 0.7, 0.9]
    return y_true, pred_proba


def test_create_logger_raise_exception(sagemaker_session) -> None:
    with pytest.raises(RuntimeError) as e:
        SagemakerExperimentsLogger(sagemaker_session=sagemaker_session[0])
    assert (
        e.value.args[0]
        == "Disable SagemakerExperimentsLogger. No current run context has "
        "been found (Failed to load a Run object. Please make sure a Run "
        "object has been initialized already.). To create a "
        "sagemaker.experiments.run explicit use experiment_name and "
        "run_name argument."
    )


def test_create_logger_explicit(sagemaker_session, mocker) -> None:
    logger = SagemakerExperimentsLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        sagemaker_session=sagemaker_session[0],
    )
    assert logger._experiment_name == EXPERIMENT_NAME
    assert logger._run_name == RUN_NAME
    assert logger._name == EXPERIMENT_NAME
    assert logger._version == RUN_NAME
    mocker.patch("sagemaker.experiments.trial_component._TrialComponent.save")
    logger.log_hyperparams({"test": "param"})
    experiments = sagemaker_session[1].list_experiments()
    assert len(experiments["ExperimentSummaries"]) == 1
    assert (
        experiments["ExperimentSummaries"][0]["ExperimentName"]
        == EXPERIMENT_NAME.lower()
    )


def test_create_logger_with_context(sagemaker_session, mocker) -> None:
    mocker.patch("sagemaker.experiments.trial_component._TrialComponent.save")
    with Run(
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        sagemaker_session=sagemaker_session[0],
    ):
        logger = SagemakerExperimentsLogger(
            sagemaker_session=sagemaker_session[0]
        )
        experiments = sagemaker_session[1].list_experiments()
        assert logger._experiment_name is None
        assert logger._run_name is None
        assert logger.name == EXPERIMENT_NAME.lower()
        assert logger.version == RUN_NAME.lower()
        assert len(experiments["ExperimentSummaries"]) == 1
        assert (
            experiments["ExperimentSummaries"][0]["ExperimentName"]
            == EXPERIMENT_NAME.lower()
        )


@pytest.mark.parametrize(
    "inp_value, out_value",
    [
        (0.1, 0.1),
        (1, 1),
        (None, "none"),
        (True, "true"),
        ("text", "text"),
        ([1, 2], [1, 2]),
        ({"1": 2}, {"1": 2}),
    ],
)
def test__prep_param_for_serialization(inp_value: Any, out_value: Any) -> None:
    assert {"value": out_value} == _prep_param_for_serialization(
        param={"value": inp_value}
    )


@pytest.mark.parametrize(
    "inp_value, out_value",
    [
        (0.1, 0.1),
        (1, 1),
        (None, "none"),
        (True, "true"),
        ("text", "text"),
        ([1, 2], [1, 2]),
        ({"1": 2}, {"1": 2}),
    ],
)
def test_log_hyperparam(
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    inp_value: Any,
    out_value: Any,
) -> None:
    logger = sme_logger[0]
    run = sme_logger[1]
    logger.log_hyperparams(params={"value": inp_value})
    assert run._trial_component.parameters["value"] == out_value


@pytest.mark.parametrize("step", [(1), (None)])
def test_log_metrics(
    step: Union[int, None],
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    mocker,
) -> None:
    # since moto does not support sagemaker metric service we have to mock the injection of the metric
    mock_log_metric = mocker.patch(
        "sagemaker.experiments._metrics._MetricsManager.log_metric"
    )
    metrics = {"F1-Score": torch.Tensor([1]), "Acc": 2.3}
    sme_logger[0].log_metrics(metrics=metrics, step=step)
    assert mock_log_metric.call_args_list[0].kwargs == {
        "metric_name": "F1-Score",
        "step": step,
        "timestamp": None,
        "value": 1.0,
    }
    assert mock_log_metric.call_args_list[1].kwargs == {
        "metric_name": "Acc",
        "step": step,
        "timestamp": None,
        "value": 2.3,
    }


@pytest.mark.parametrize(
    "title, is_output, no_skill, pos_label",
    [
        ("my-title", True, 2, None),
        ("my-title", True, 2, 0),
        (None, False, None, None),
    ],
)
def test_log_precision_recall(
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    title: Union[str, None],
    is_output: bool,
    no_skill: Union[int, None],
    pos_label: Union[int, None],
    binary_labels: Tuple[List, List],
    mocker,
) -> None:
    y_true, pred_proba = binary_labels
    # since create_artifact has not implemented by moto we have to mock the sagemaker function
    mock_func = mocker.patch("sagemaker.experiments.Run.log_precision_recall")

    sme_logger[0].log_precision_recall(
        y_true=y_true,
        predicted_probabilities=pred_proba,
        positive_label=pos_label,
        title=title,
        is_output=is_output,
        no_skill=no_skill,
    )
    assert mock_func.call_args.kwargs == {
        "y_true": y_true,
        "predicted_probabilities": pred_proba,
        "positive_label": pos_label,
        "title": title,
        "is_output": is_output,
        "no_skill": no_skill,
    }


@pytest.mark.parametrize(
    "title, is_output",
    [
        ("my-title", True),
        (None, False),
    ],
)
def test_log_roc_curve(
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    title: Union[str, None],
    is_output: bool,
    binary_labels: Tuple[np.ndarray, np.ndarray],
    mocker,
) -> None:
    y_true, pred_proba = binary_labels
    # since create_artifact has not implemented by moto we have to mock the sagemaker function
    mock_func = mocker.patch("sagemaker.experiments.Run.log_roc_curve")
    sme_logger[0].log_roc_curve(
        y_true=y_true,
        y_score=pred_proba,
        title=title,
        is_output=is_output,
    )
    assert mock_func.call_args.kwargs == {
        "y_true": y_true,
        "y_score": pred_proba,
        "title": title,
        "is_output": is_output,
    }


@pytest.mark.parametrize(
    "title, is_output",
    [
        ("my-title", True),
        (None, False),
    ],
)
def test_log_confusion_matrix(
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    title: Union[str, None],
    is_output: bool,
    binary_labels: Tuple[np.ndarray, np.ndarray],
    mocker,
) -> None:
    y_true, pred_proba = binary_labels
    # since create_artifact has not implemented by moto we have to mock the sagemaker function
    mock_func = mocker.patch("sagemaker.experiments.Run.log_confusion_matrix")
    sme_logger[0].log_confusion_matrix(
        y_true=y_true,
        y_pred=pred_proba,
        title=title,
        is_output=is_output,
    )
    assert mock_func.call_args.kwargs == {
        "y_true": y_true,
        "y_pred": pred_proba,
        "title": title,
        "is_output": is_output,
    }


@pytest.mark.parametrize(
    "media_type, is_output",
    [("text/plain", True), (None, False)],
)
def test_log_artifact(
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    media_type: Union[str, None],
    is_output: bool,
) -> None:
    artefact_name = "TestArtefact"
    artefact_value = "TestValue"
    sme_logger[0].log_artifact(
        name=artefact_name,
        is_output=is_output,
        value=artefact_value,
        media_type=media_type,
    )
    artefact = (
        sme_logger[1]._trial_component.output_artifacts
        if is_output
        else sme_logger[1]._trial_component.input_artifacts
    )
    assert artefact[artefact_name].media_type == media_type
    assert artefact[artefact_name].value == artefact_value


@pytest.mark.parametrize(
    "media_type, is_output",
    [("text/plain", True), (None, False)],
)
def test_log_file(
    sme_logger: Tuple[SagemakerExperimentsLogger, Run],
    media_type: Union[str, None],
    is_output: bool,
    mocker,
) -> None:
    artefact_name = "TestArtefact"
    file_path = "dumpfile.csv"
    s3_uri = "s3://testbucket/dumpfile.csv"
    mocker.patch.object(
        sme_logger[1]._artifact_uploader,
        "upload_artifact",
        return_value=(s3_uri, None),
    )

    sme_logger[0].log_file(
        file_path=file_path,
        name=artefact_name,
        media_type=media_type,
        is_output=is_output,
    )
    artefact = (
        sme_logger[1]._trial_component.output_artifacts
        if is_output
        else sme_logger[1]._trial_component.input_artifacts
    )
    assert (
        artefact[artefact_name].media_type == media_type
        if media_type
        else "text/csv"
    )
    assert artefact[artefact_name].value == s3_uri


def test_name(sme_logger: Tuple[SagemakerExperimentsLogger, Run]) -> None:
    sme_logger[0].name == EXPERIMENT_NAME


def test_version(sme_logger: Tuple[SagemakerExperimentsLogger, Run]) -> None:
    sme_logger[0].version == RUN_NAME
