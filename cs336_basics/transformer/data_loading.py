import torch
import numpy.typing as npt


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 先转为torch.Tensor以便后续操作
    dataset = torch.Tensor(dataset)
    # 获取总长度，dataset为一维张量
    total_length = dataset.shape[0]
    # 最大的起始索引, 需要再 -1, 因为要防止label越界
    max_start_index = total_length - context_length - 1
    start_indices = torch.randint(low=0, high=max_start_index, size=(batch_size,))
    data_batch = torch.zeros((batch_size, context_length))
    label_batch = torch.zeros((batch_size, context_length))

    for i, start_index in enumerate(start_indices):
        data_batch[i] = dataset[start_index : start_index + context_length]
        label_batch[i] = dataset[start_index + 1 : start_index + context_length + 1]

    return data_batch.to(device), label_batch.to(device)
