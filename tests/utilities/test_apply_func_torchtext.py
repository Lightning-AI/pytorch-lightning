import torch
import torchtext
from torchtext.data.example import Example

from pytorch_lightning.utilities.apply_func import move_data_to_device


def _get_torchtext_data_iterator(include_lengths=False):
    text_field = torchtext.data.Field(sequential=True, pad_first=False,  # nosec
                                      init_token="<s>", eos_token="</s>",  # nosec
                                      include_lengths=include_lengths)  # nosec

    example1 = Example.fromdict({"text": "a b c a c"}, {"text": ("text", text_field)})
    example2 = Example.fromdict({"text": "b c a a"}, {"text": ("text", text_field)})
    example3 = Example.fromdict({"text": "c b a"}, {"text": ("text", text_field)})

    dataset = torchtext.data.Dataset([example1, example2, example3],
                                     {"text": text_field}
                                     )
    text_field.build_vocab(dataset)

    iterator = torchtext.data.Iterator(dataset, batch_size=3,
                                       sort_key=None, device=None,
                                       batch_size_fn=None,
                                       train=True, repeat=False, shuffle=None,
                                       sort=None, sort_within_batch=None)
    return iterator, text_field


def test_batch_move_data_to_device_torchtext_include_lengths_false():
    cuda_device_cnt = torch.cuda.device_count()
    if cuda_device_cnt > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_iterator, _ = _get_torchtext_data_iterator(include_lengths=True)
    data_iter = iter(data_iterator)
    batch = next(data_iter)

    # this call should not throw an error
    batch_on_device = move_data_to_device(batch, device)
    # tensor with data
    assert (batch_on_device.text[0].device == device)
    # tensor with length of data
    assert (batch_on_device.text[1].device == device)


def test_batch_move_data_to_device_torchtext_include_lengths_true():
    cuda_device_cnt = torch.cuda.device_count()
    if cuda_device_cnt > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_iterator, _ = _get_torchtext_data_iterator(include_lengths=False)
    data_iter = iter(data_iterator)
    batch = next(data_iter)

    # this call should not throw an error
    batch_on_device = move_data_to_device(batch, device)
    # tensor with data
    assert (batch_on_device.text[0].device == device)
