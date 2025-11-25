import torch
import numpy.typing as npt


def data_loading(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 获取总长度，dataset为一维张量
    total_length = dataset.shape[0]
    # 最大的起始索引
    max_start_index = total_length - context_length
    start_indices = torch.randint(low=0, high=max_start_index, size=(batch_size,))
    # 提前分配内存
    data_batch = torch.zeros((batch_size, context_length), dtype=dtype)
    label_batch = torch.zeros((batch_size, context_length), dtype=dtype)

    for i, start_index in enumerate(start_indices):
        # 逐个转化为Tensor
        data_batch[i] = torch.Tensor(
            dataset[start_index : start_index + context_length]
        )
        label_batch[i] = torch.Tensor(
            dataset[start_index + 1 : start_index + context_length + 1]
        )

    return data_batch.to(device), label_batch.to(device)
