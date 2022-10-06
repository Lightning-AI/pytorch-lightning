from unittest import mock

import pytest

from lightning_lite.plugins.collectives import SingleDeviceCollective


@pytest.fixture(autouse=True)
def check_destroy_group():
    with mock.patch(
        "lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.new_group",
        wraps=SingleDeviceCollective.new_group,
    ) as mock_new, mock.patch(
        "lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.destroy_group",
        wraps=SingleDeviceCollective.destroy_group,
    ) as mock_destroy:
        yield
        assert (
            mock_new.call_count == mock_destroy.call_count
        ), "new_group and destroy_group should be called the same number of times"


def test_can_instantiate():
    SingleDeviceCollective()


def test_create_group():
    collective = SingleDeviceCollective(init_kwargs={"arg1": None, "arg2": 10}, group_kwargs={"arg3": None, "arg4": 10})
    with pytest.raises(RuntimeError, match=r"SingleDeviceCollective does not own a group\. HINT:"):
        _ = collective.group

    with mock.patch(
        "lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.init_group"
    ) as init_mock, mock.patch(
        "lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.new_group"
    ) as new_mock:
        collective.create_group(init_kwargs={"arg2": 13, "arg3": 10}, group_kwargs={"arg1": 15, "arg3": 10})
    init_kwargs = {"arg1": None, "arg2": 13, "arg3": 10}
    group_kwargs = {"arg3": 10, "arg4": 10, "arg1": 15}

    init_mock.assert_called_once_with(**init_kwargs)
    new_mock.assert_called_once_with(**group_kwargs)
    assert collective._init_kwargs == init_kwargs
    assert collective._group_kwargs == group_kwargs

    with mock.patch("lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.destroy_group"):
        collective.teardown()
