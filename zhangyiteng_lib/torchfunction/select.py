import torch

def select_index_by_batch(x: torch.Tensor, index: torch.Tensor, batch_dim: int=0, selected_dim: int=1) -> torch.Tensor:
    """
    input:
        x: torch.Tensor[batch, selected, ...]
        index: torch.LongTensor[batch]

    ret:
        torch.Tensor[batch, ...]
    """
    shape = x.shape
    batch_dim = batch_dim % len(shape)
    selected_dim = selected_dim % len(shape)
    assert batch_dim != selected_dim
    batch_size = shape[batch_dim]
    ones = [1 for _ in range(len(shape))]
    ones[batch_dim] = batch_size
    _shape = list(shape)
    _shape[selected_dim] = 1

    return torch.gather(x, selected_dim, index.view(*ones).expand(*_shape)).squeeze(selected_dim)
