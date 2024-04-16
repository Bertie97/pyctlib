import torch
from typing import Optional, Dict, KeysView

class RunningAverage:
    def __init__(self, default_dim: int = 0, device: Optional[str]=None):
        """
        Initializes an instance of AverageGather.

        Args:
            default_dim (int): The default dimension along which to calculate the mean.
        """
        self.__content: Dict[str, torch.Tensor] = {}
        self.__content_num: Dict[str, torch.Tensor] = {}
        self.__content_pow2_sum: Dict[str, int] = {}
        self.__default_dim: int = default_dim
        self.__device = device

    def update(self, key: str, value: torch.Tensor, dim: Optional[int] = None) -> None:
        """
        Updates the running average statistics for a given key.

        Args:
            key (str): The key to identify the data.
            value (torch.Tensor): The input data tensor.
            dim (Optional[int]): The dimension along which to calculate the mean.
                If None, uses the default_dim.

        Raises:
            TypeError: If value is not a torch.Tensor.
            ValueError: If the specified dimension (dim) is out of range for the input tensor.
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("Input 'value' must be a torch.Tensor.")

        if dim is not None and (dim < 0 or dim >= len(value.shape)):
            raise ValueError("Specified dimension 'dim' is out of range for the input tensor.")

        dim = dim if dim is not None else self.__default_dim

        device = self.__device if self.__device is not None else value.device
        if self.__device is not None:
            value = value.to(device)

        if key not in self.__content:
            self.__content[key] = value.mean(dim=dim)
            self.__content_num[key] = value.shape[dim]
            self.__content_pow2_sum[key] = value.pow(2).sum(dim=dim)
        else:
            if value.mean(dim).shape != self.__content[key].shape:
                raise ValueError("Input tensor dimension does not match the existing data for the key. Existing shape:", self.__content[key].shape, " Input shape:", value.mean(dim).shape)
            current_mean = self.__content[key]
            current_num = self.__content_num[key]
            current_pow2_sum = self.__content_pow2_sum[key]

            new_sum = current_mean * current_num + value.sum(dim=dim)
            new_num = current_num + value.shape[dim]

            self.__content[key] = new_sum / new_num
            self.__content_num[key] = new_num
            self.__content_pow2_sum[key] = current_pow2_sum + value.pow(2).sum(dim=dim)

    def keys(self) -> KeysView[str]:
        """
        Returns the keys of the stored data.

        Returns:
            KeysView[str]: The keys of the stored data.
        """
        return self.__content.keys()

    def get_content(self) -> Dict[str, torch.Tensor]:
        """
        Returns the dictionary containing the mean values for each key.

        Returns:
            Dict[str, torch.Tensor]: The dictionary containing mean values for each key.
        """
        return self.__content.copy()

    def mean(self, key: str) -> torch.Tensor:
        """
        Gets the mean value for a given key.

        Args:
            key (str): The key to retrieve the mean value.

        Returns:
            torch.Tensor: The mean value corresponding to the key.
        """
        return self.__content[key]

    def variance(self, key: str) -> torch.Tensor:
        """
        Calculates the variance for a given key.

        Args:
            key (str): The key to calculate the variance.

        Returns:
            torch.Tensor: The variance value corresponding to the key.
        """
        pow2_sum = self.__content_pow2_sum[key]
        mean_pow2 = self.__content[key].pow(2)

        return pow2_sum / self.__content_num[key] - mean_pow2

    def count(self, key: str) -> int:
        """
        Gets the total count of elements for a given key.

        Args:
            key (str): The key to retrieve the count.

        Returns:
            int: The total count of elements for the key.
        """
        return self.__content_num[key]
