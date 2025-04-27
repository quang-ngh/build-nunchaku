import typing as tp
import os 
import torch
import safetensors

from huggingface_hub import hf_hub_download


def fetch_or_download(path, repo_type="model"):
    if not os.path.exists(path):
        hf_repo_id = os.path.dirname(path)
        filename = os.path.basename(path)
        path = hf_hub_download(repo_id=hf_repo_id, filename=filename, repo_type=repo_type)
    return path
def ceil_divide(x, divisor):
    """Ceiling division.

    Args:
        x (`int`):
            dividend.
        divisor (`int`):
            divisor.

    Returns:
        `int`:
            ceiling division result.
    """
    return (x + divisor - 1) // divisor


def load_state_dict_in_safetensors(
    path, device, filter_prefix=""
) :
    """Load state dict in SafeTensors.

    Args:
        path (`str`):
            file path.
        device (`str` | `torch.device`, optional, defaults to `"cpu"`):
            device.
        filter_prefix (`str`, optional, defaults to `""`):
            filter prefix.

    Returns:
        `dict`:
            loaded SafeTensors.
    """
    state_dict = {}
    with safetensors.safe_open(fetch_or_download(path), framework="pt", device=device) as f:
        for k in f.keys():
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    return state_dict

def is_nunchaku_format(lora):
    if isinstance(lora, str):
        tensors = load_state_dict_in_safetensors(lora, device="cpu")
    else:
        tensors = lora

    for k in tensors.keys():
        if ".mlp_fc" in k or "mlp_context_fc1" in k:
            return True
    return False


def pad(
    tensor,
    divisor,
    dim,
    fill_value=0,
) :
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result
