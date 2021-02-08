import pytest
import torch

from pytorch_lightning.metrics.functional.image_gradients import image_gradients


def test_invalid_input_img_type():
    """Test Whether the module successfully handles invalid input data type"""
    invalid_dummy_input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    with pytest.raises(TypeError):
        image_gradients(invalid_dummy_input)


def test_invalid_input_ndims():
    """
    Test whether the module successfully handles invalid number of dimensions
    of input tensor
    """

    BATCH_SIZE = 1
    HEIGHT = 5
    WIDTH = 5
    CHANNELS = 1

    image = torch.arange(0, BATCH_SIZE * HEIGHT * WIDTH * CHANNELS, dtype=torch.float32)
    image = torch.reshape(image, (HEIGHT, WIDTH))

    with pytest.raises(RuntimeError):
        image_gradients(image)


def test_multi_batch_image_gradients():
    """Test whether the module correctly calculates gradients for known input
    with non-unity batch size.Example input-output pair taken from TF's implementation of i
    mage-gradients
    """

    BATCH_SIZE = 5
    HEIGHT = 5
    WIDTH = 5
    CHANNELS = 1

    single_channel_img = torch.arange(0, 1 * HEIGHT * WIDTH * CHANNELS, dtype=torch.float32)
    single_channel_img = torch.reshape(single_channel_img, (CHANNELS, HEIGHT, WIDTH))
    image = torch.stack([single_channel_img for _ in range(BATCH_SIZE)], dim=0)

    true_dy = [
        [5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5.],
        [0., 0., 0., 0., 0.],
    ]

    true_dx = [
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
    ]
    true_dy = torch.Tensor(true_dy)
    true_dx = torch.Tensor(true_dx)

    dy, dx = image_gradients(image)

    for batch_id in range(BATCH_SIZE):
        assert torch.allclose(dy[batch_id, 0, :, :], true_dy)
    assert dy.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)
    assert dx.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)


def test_image_gradients():
    """Test whether the module correctly calculates gradients for known input.
    Example input-output pair taken from TF's implementation of image-gradients
    """

    BATCH_SIZE = 1
    HEIGHT = 5
    WIDTH = 5
    CHANNELS = 1

    image = torch.arange(0, BATCH_SIZE * HEIGHT * WIDTH * CHANNELS, dtype=torch.float32)
    image = torch.reshape(image, (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))

    true_dy = [
        [5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5.],
        [5., 5., 5., 5., 5.],
        [0., 0., 0., 0., 0.],
    ]

    true_dx = [
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 0.],
    ]

    true_dy = torch.Tensor(true_dy)
    true_dx = torch.Tensor(true_dx)

    dy, dx = image_gradients(image)

    assert torch.allclose(dy, true_dy), "dy fails test"
    assert torch.allclose(dx, true_dx), "dx fails tests"
