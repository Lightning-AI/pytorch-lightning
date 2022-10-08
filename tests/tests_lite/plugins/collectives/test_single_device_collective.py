from unittest import mock

import pytest

from lightning_lite.plugins.collectives import SingleDeviceCollective


def test_can_instantiate_without_args():
    SingleDeviceCollective()


def test_create_group():
    collective = SingleDeviceCollective()
    assert collective.is_available()
    assert collective.is_initialized()

    with pytest.raises(RuntimeError, match=r"SingleDeviceCollective` does not own a group"):
        _ = collective.group

    with mock.patch(
        "lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.new_group"
    ) as new_mock:
        collective.create_group(arg1=15, arg3=10)

    group_kwargs = {"arg3": 10, "arg1": 15}
    new_mock.assert_called_once_with(**group_kwargs)

    with mock.patch("lightning_lite.plugins.collectives.single_device_collective.SingleDeviceCollective.destroy_group"):
        collective.teardown()
