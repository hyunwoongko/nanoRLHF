import importlib
import inspect
from dataclasses import asdict, is_dataclass, fields, MISSING
from typing import Any

import torch
import torch.distributed as dist

from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU

NoneType = type(None)


def is_hashable(x) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False


def current_device():
    """
    Get the current CUDA device.

    Returns:
        torch.device: The current CUDA device.
    """
    return torch.device(torch.cuda.current_device())


def resolve_qualname(module_name: str, qualname: str) -> Any:
    """
    Resolve a qualified name to an object.

    Args:
        module_name (str): The module name.
        qualname (str): The qualified name within the module.

    Returns:
        Any: The resolved object.
    """
    module = importlib.import_module(module_name)
    data = module
    for attr in qualname.split("."):
        data = getattr(data, attr)
    return data


def get_object_without_init(cls: type) -> Any:
    """
    Create an instance of a class without calling its __init__ method.

    Args:
        cls (type): The class to instantiate.

    Returns:
        Any: An instance of the class.
    """
    try:
        return cls.__new__(cls)  # noqa
    except TypeError:
        return object.__new__(cls)  # noqa


def setattr_frozen_safe(instance: Any, name: str, value: Any):
    """
    Safely set an attribute on a potentially frozen dataclass instance.

    Args:
        instance (Any): The instance to modify.
        name (str): The name of the attribute to set.
        value (Any): The value to set the attribute to.
    """
    try:
        object.__setattr__(instance, name, value)
    except Exception:
        setattr(instance, name, value)


def pack_object_supported(data: Any) -> dict:
    """
    Pack an object into a dictionary for communication if it's of a supported type.

    Args:
        data (Any): The object to pack.

    Returns:
        dict: The packed object dictionary.
    """
    state = {}
    for name in dir(data):
        if not name or name.startswith("_"):
            continue
        try:
            value = getattr(data, name)
        except Exception:
            continue
        if callable(value):
            continue
        state[name] = value
    return state


class P2P:
    """
    Peer-to-peer communication handler for sending and receiving various data types
    between different ranks in a distributed setting.

    Examples:
        >>> import torch
        >>> p2p = P2P()
        >>> # Define a complex nested data structure to send
        >>> data = {
        ...     ["key": "value", "number": 42, "tensor": torch.tensor([1, 2, 3])],
        ...     ("tuple", 3.14, False),
        ...     "Just a string",
        ...     12345,
        ...     None,
        ...     {1, 2, 3},
        ... }
        >>> # Use your MPU and ParallelMode instances here
        >>> mpu, mode = ...
        >>> # Try to send from rank 0 to rank 1
        >>> if torch.distributed.get_rank() == 0:
        ...     dst_rank = 1
        ...     p2p.send(data, dst_rank, mpu, mode)
        >>> elif torch.distributed.get_rank() == 1:
        ...     src_rank = 0
        ...     received_data = p2p.recv(src_rank, mpu, mode)
        ...     print(received_data == data)  # True

    Discussion:
        Q. What data types are supported?
            Supported types include `bool`, `int`, `float`, `complex`, `str`, `type`,
            `list`, `tuple`, `set`, `dict`, `NoneType`, `torch.Size`, `torch.Tensor`,
            `dataclass` instances, and `Cache` objects from the `transformers` library.
            Nested structures of these types such as list of dicts are also supported.

        Q. How to send various data types using torch.distributed.send/recv?
            In `send`, we first sanitize data into supported forms, then dispatch to a
            type-specific `_send_*` routine via a mapping. Each routine sends a small
            header (metadata) followed by the payload. For strings, we send the length and
            then the Unicode code points (not ASCII). For tensors, we send dtype id,
            requires_grad flag, rank, shape, and then the contiguous tensor buffer.

            The `recv` side first receives the type id and then calls the matching
            `_recv_*` to reconstruct the original object.
            (i.e., the sender explicitly transmits a type token before the payload.)

        Q. How to send dataclass and custom objects like `transformers.cache_utils.Cache`?
            For dataclasses, we serialize with `asdict` and include module/qualname so the
            receiver can locate the class. Reconstruction prefers calling `__init__` with
            available fields; if that’s not possible, we safely bypass `__init__` and set
            fields directly (handles frozen/slots cases).

            For Transformers cache objects, we don’t pickle arbitrary objects. Instead,
            we package them in a dedicated format (`__kind__="hf_cache"`) including the
            class meta (module/qualname) and only those public attributes that our P2P
            layer can transmit (version/name changes tolerated). The receiver recreates
            the instance without `__init__` and restores attributes. Arbitrary custom
            objects beyond these are intentionally out of scope for performance/safety.
    """

    def __init__(self, mpu: MPU, mode: ParallelMode = ParallelMode.PIPELINE):
        from nanorlhf.nanotron.core.pp.loss import MicroLossTensor

        self.mpu = mpu
        self.mode = mode
        self.group = mpu.get_group(mode)

        self.torch_id_to_dtype = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        self.id_to_dtype = [
            bool,
            int,
            float,
            complex,
            str,
            type,
            list,
            tuple,
            set,
            dict,
            NoneType,
            torch.Size,
            torch.Tensor,
            MicroLossTensor,
        ]
        self.supported_atoms = (
            bool,
            int,
            float,
            complex,
            str,
            type,
            NoneType,
            torch.Size,
            torch.Tensor,
            MicroLossTensor,
        )
        self.dtype_to_id = {dtype: idx for idx, dtype in enumerate(self.id_to_dtype)}
        self.torch_dtype_to_id = {dtype: idx for idx, dtype in enumerate(self.torch_id_to_dtype)}

        self.instructions = {
            bool: {"send": self.send_bool, "recv": self.recv_bool},
            int: {"send": self.send_int, "recv": self.recv_int},
            float: {"send": self.send_float, "recv": self.recv_float},
            complex: {"send": self.send_complex, "recv": self.recv_complex},
            str: {"send": self.send_str, "recv": self.recv_str},
            type: {"send": self.send_type, "recv": self.recv_type},
            list: {"send": self.send_list, "recv": self.recv_list},
            tuple: {"send": self.send_tuple, "recv": self.recv_tuple},
            set: {"send": self.send_set, "recv": self.recv_set},
            dict: {"send": self.send_dict, "recv": self.recv_dict},
            NoneType: {"send": self.send_none, "recv": self.recv_none},
            torch.Size: {"send": self.send_size, "recv": self.recv_size},
            torch.Tensor: {"send": self.send_tensor, "recv": self.recv_tensor},
            MicroLossTensor: {"send": self.send_tensor, "recv": self.recv_tensor},
        }

        self._huggingface_cache_classes = set()
        self.support_huggingface_cache = self.enable_huggingface_cache_support()
        # dataclass object and huggingface cache use `send_dict` and `recv_dict`
        # after packing/unpacking to/from a dict type.

    def enable_huggingface_cache_support(self):
        """
        Enable support for huggingface cache objects.

        Returns:
            bool: True if huggingface cache support is enabled, False otherwise.
        """
        try:
            from transformers import cache_utils
        except Exception:
            return False

        ok = False
        for name, cls in inspect.getmembers(cache_utils, inspect.isclass):
            if not name or name.startswith("_") or not name.endswith("Cache"):
                continue
            if getattr(cls, "__module__", "") != cache_utils.__name__:
                continue

            self._huggingface_cache_classes.add(cls)
            self.instructions[cls] = {"send": self.send_cache, "recv": self.recv_cache}
            ok = True

        return ok

    def is_huggingface_cache(self, data: Any) -> bool:
        """
        Check if the data is an instance of a huggingface cache class.

        Args:
            data (Any): The data to check.

        Returns:
            bool: True if the data is a huggingface cache instance, False otherwise.
        """
        if any(isinstance(data, c) for c in self._huggingface_cache_classes):
            return True
        return hasattr(data, "key_cache") and hasattr(data, "value_cache")

    def pack_pyset(self, data: set) -> dict:
        """
        Pack a Python set into a dict payload, so it can survive sanitize/transport.
        """
        return {
            "__kind__": "py_set",
            "items": [self.sanitize_for_p2p(x) for x in data],  # 각 원소 sanitize
        }

    def pack_dataclass(self, data) -> dict:
        """
        Pack a dataclass object into a dictionary for communication.

        Args:
            data (Any): The dataclass object to pack.

        Returns:
            dict: The packed dataclass dictionary.
        """
        return {
            "__kind__": "dataclass",
            "module": data.__class__.__module__,
            "qualname": data.__class__.__qualname__,
            "state": self.sanitize_for_p2p(asdict(data)),
        }

    def pack_cache(self, data: Any):
        """
        Pack a huggingface cache object into a dictionary for communication.

        Args:
            data (Any): The huggingface cache object to pack.

        Returns:
            dict: The packed cache dictionary.
        """
        return {
            "__kind__": "hf_cache",
            "module": data.__class__.__module__,
            "qualname": data.__class__.__qualname__,
            "state": self.sanitize_for_p2p(pack_object_supported(data)),
        }

    def maybe_reconstruct_special(self, data: Any) -> Any:
        """
        Reconstruct special objects like dataclasses and huggingface cache.

        Args:
            data (Any): The object to potentially reconstruct.

        Returns:
            Any: The reconstructed object if applicable, otherwise the original object.
        """
        if not (isinstance(data, dict) and "__kind__" in data):
            return data

        if data.get("__kind__") == "py_set":
            items = [self.maybe_reconstruct_special(item) for item in data.get("items", [])]
            output = set()
            for item in items:
                if is_hashable(item):
                    output.add(item)
                else:
                    raise TypeError(
                        f"Unhashable element inside set during P2P reconstruction: "
                        f"{type(item).__name__} -> {item!r}"
                    )
            return output
        else:
            cls = resolve_qualname(data["module"], data["qualname"])
            instance = get_object_without_init(cls)
            state = data.get("state", {})
            for k, v in state.items():
                try:
                    setattr_frozen_safe(instance, k, self.maybe_reconstruct_special(v))
                except Exception:
                    pass
            return instance

    def sanitize_for_p2p(self, data: Any) -> Any:
        """
        Sanitize data to be in a supported form for peer-to-peer communication.

        Args:
            data (Any): The data to sanitize.

        Returns:
            Any: The sanitized data.
        """
        if isinstance(data, self.supported_atoms):
            return data
        if isinstance(data, set):
            return self.pack_pyset(data)
        if is_dataclass(data):
            return self.pack_dataclass(data)
        if self.support_huggingface_cache and self.is_huggingface_cache(data):
            return self.pack_cache(data)
        if isinstance(data, (list, tuple)):
            return type(data)(self.sanitize_for_p2p(d) for d in data)
        if isinstance(data, dict):
            output = {}
            for k, v in data.items():
                sanitized_k = self.sanitize_for_p2p(k)
                sanitized_v = self.sanitize_for_p2p(v)
                if sanitized_k is None:
                    continue
                if is_hashable(sanitized_k):
                    output[sanitized_k] = sanitized_v
                else:
                    raise TypeError(
                        f"Unhashable dict key after sanitize: " f"{type(sanitized_k).__name__} -> {sanitized_k!r}"
                    )
            return output
        return None

    def send(self, data, dst_rank: int):
        """
        Send data to the destination rank.

        Args:
            data: The data to send. Supported types are bool, int, float, complex, str, type,
                  list, tuple, set, dict, NoneType, torch.Size, torch.Tensor.
            dst_rank (int): The destination rank to send the data to.
        """
        data = self.sanitize_for_p2p(data)
        _type = type(data)
        assert _type in self.id_to_dtype, f"unsupported type: {_type}"
        self.instructions[_type]["send"](data, dst_rank=dst_rank, send_type=True)  # noqa

    def recv(self, src_rank: int):
        """
        Receive data from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            The received data. Supported types are bool, int, float, complex, str, type,
            list, tuple, set, dict, NoneType, torch.Size, torch.Tensor.
        """
        _type = self.instructions[type]["recv"](src_rank=src_rank)  # noqa
        assert _type in self.id_to_dtype, f"unsupported type: {_type}"
        return self.instructions[_type]["recv"](src_rank=src_rank)  # noqa

    def send_type(self, data: type, dst_rank: int, send_type: bool = False):
        """
        Send a type to the destination rank.

        Args:
            data (type): The type to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, type), f"Wrong type: {data} must be {type} type."
        tensor = torch.tensor([self.dtype_to_id[data]], dtype=torch.long, device=current_device())
        dist.send(tensor, dst=dst_rank, group=self.group)

    def recv_type(self, src_rank: int) -> type:
        """
        Receive a type from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.
        """
        tensor = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(tensor, src=src_rank, group=self.group)
        return self.id_to_dtype[tensor.item()]

    def send_none(self, data: NoneType, dst_rank: int, send_type: bool = False):
        """
        Send nothing, just assert data is None.

        Args:
            data (NoneType): The data to send, must be None.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, NoneType), f"Wrong type: {data} must be {NoneType}."
        if send_type:
            self.send_type(NoneType, dst_rank)

    def recv_none(self, src_rank: int):
        """
        Just return None.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            NoneType: None.
        """
        return None

    def send_str(self, data: str, dst_rank: int, send_type: bool = False):
        """
        Send a string to the destination rank.

        Args:
            data (str): The string to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, str), f"Wrong type: {data} must be {str}."

        if send_type is True:
            self.send_type(str, dst_rank=dst_rank)

        length = torch.tensor([len(data)], dtype=torch.long, device=current_device())
        dist.send(length, dst=dst_rank, group=self.group)

        payload = torch.tensor([ord(s) for s in data], dtype=torch.long, device=current_device())
        dist.send(payload, dst=dst_rank, group=self.group)

    def recv_str(self, src_rank: int) -> str:
        """
        Receive a string from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            str: The received string.
        """
        length = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(length, src=src_rank, group=self.group)
        payload = torch.empty(length.item(), dtype=torch.long, device=current_device())
        dist.recv(payload, src=src_rank, group=self.group)
        return "".join([chr(i) for i in payload.tolist()])

    def send_bool(self, data: bool, dst_rank: int, send_type: bool = False):
        """
        Send a boolean value to the destination rank.

        Args:
            data (bool): The boolean value to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, bool), f"Wrong type: {data} must be {bool}."

        if send_type is True:
            self.send_type(bool, dst_rank=dst_rank)

        tensor = torch.tensor([1 if data else 0], dtype=torch.long, device=current_device())
        dist.send(tensor, dst=dst_rank, group=self.group)

    def recv_bool(self, src_rank: int) -> bool:
        """
        Receive a boolean value from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            bool: The received boolean value.
        """
        tensor = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(tensor, src=src_rank, group=self.group)
        value = tensor.item()
        if value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise ValueError(f"Wrong value for boolean. only 0 or 1 can be supported. " f"but your input is {value}.")

    def send_int(self, data: int, dst_rank: int, send_type: bool = False):
        """
        Send an integer to the destination rank.

        Args:
            data (int): The integer to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, int), f"Wrong type: {data} must be {int}."

        if send_type is True:
            self.send_type(int, dst_rank=dst_rank)

        tensor = torch.tensor([data], dtype=torch.long, device=current_device())
        dist.send(tensor, dst=dst_rank, group=self.group)

    def recv_int(self, src_rank: int) -> int:
        """
        Receive an integer from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            int: The received integer.
        """
        tensor = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(tensor, src=src_rank, group=self.group)
        return tensor.item()

    def send_float(self, data: float, dst_rank: int, send_type: bool = False):
        """
        Send a float to the destination rank.

        Args:
            data (float): The float to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, float), f"Wrong type: {data} must be {float}."

        if send_type is True:
            self.send_type(float, dst_rank=dst_rank)

        tensor = torch.tensor([data], dtype=torch.float32, device=current_device())
        dist.send(tensor, dst=dst_rank, group=self.group)

    def recv_float(self, src_rank: int) -> float:
        """
        Receive a float from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            float: The received float.
        """
        tensor = torch.tensor([0.0], dtype=torch.float32, device=current_device())
        dist.recv(tensor, src=src_rank, group=self.group)
        return tensor.item()

    def send_complex(self, data: complex, dst_rank: int, send_type: bool = False):
        """
        Send a complex number to the destination rank.

        Args:
            data (complex): The complex number to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, complex), f"Wrong type: {data} must be {complex}."

        if send_type is True:
            self.send_type(complex, dst_rank=dst_rank)

        tensor = torch.tensor([data.real, data.imag], dtype=torch.float32, device=current_device())
        dist.send(tensor, dst=dst_rank, group=self.group)

    def recv_complex(self, src_rank: int) -> complex:
        """
        Receive a complex number from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            complex: The received complex number.
        """
        tensor = torch.tensor([0.0, 0.0], dtype=torch.float32, device=current_device())
        dist.recv(tensor, src=src_rank, group=self.group)
        return complex(tensor[0].item(), tensor[1].item())

    def send_tensor(self, data: torch.Tensor, dst_rank: int, send_type: bool = False):
        """
        Send a tensor to the destination rank.

        Args:
            data (torch.Tensor): The tensor to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, torch.Tensor), f"Wrong type: {data} must be {torch.Tensor}."

        if send_type is True:
            self.send_type(torch.Tensor, dst_rank=dst_rank)

        dtype = torch.tensor(self.torch_dtype_to_id[data.dtype], dtype=torch.long, device=current_device())
        dist.send(dtype, dst=dst_rank, group=self.group)

        requires_grad = torch.tensor(1 if data.requires_grad else 0, dtype=torch.long, device=current_device())
        dist.send(requires_grad, dst=dst_rank, group=self.group)

        dims = torch.tensor(len(data.size()), dtype=torch.long, device=current_device())
        dist.send(dims, dst=dst_rank, group=self.group)

        shape = torch.tensor(list(data.size()), dtype=torch.long, device=current_device())
        dist.send(shape, dst=dst_rank, group=self.group)

        if not data.is_contiguous():
            data = data.contiguous()

        dist.send(data, dst=dst_rank, group=self.group)

    def recv_tensor(self, src_rank: int) -> torch.Tensor:
        """
        Receive a tensor from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            torch.Tensor: The received tensor.
        """
        dtype = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(dtype, src=src_rank, group=self.group)
        dtype = self.torch_id_to_dtype[dtype.item()]

        requires_grad = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(requires_grad, src=src_rank, group=self.group)
        requires_grad = True if requires_grad.item() == 1 else False

        dims = torch.tensor([0], dtype=torch.long, device=current_device())
        dist.recv(dims, src=src_rank, group=self.group)
        dims = dims.item()

        shape = torch.tensor([0] * dims, dtype=torch.long, device=current_device())
        dist.recv(shape, src=src_rank, group=self.group)
        shape = tuple(shape.tolist())

        data = torch.zeros(size=shape, dtype=dtype, device=current_device())
        data.requires_grad = requires_grad and data.is_floating_point()
        dist.recv(data, src=src_rank, group=self.group)
        return data

    def send_list(self, data: list, dst_rank: int, send_type: bool = False):
        """
        Send a list to the destination rank.

        Args:
            data (list): The list to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, list), f"wrong type: {data} must be {list} type."

        if send_type is True:
            self.send_type(list, dst_rank=dst_rank)

        length = len(data)
        self.send_int(length, dst_rank=dst_rank)

        for item in data:
            _type = type(item)
            assert _type in self.id_to_dtype, f"unsupported type: {_type}"
            self.instructions[_type]["send"](item, dst_rank=dst_rank, send_type=True)  # noqa

    def recv_list(self, src_rank: int) -> list:
        """
        Receive a list from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            list: The received list.
        """
        output_list = []

        length = self.recv_int(src_rank=src_rank)

        for _ in range(length):
            _type = self.instructions[type]["recv"](src_rank=src_rank)  # noqa
            assert _type in self.id_to_dtype, f"unsupported type: {_type}"
            _item = self.instructions[_type]["recv"](src_rank=src_rank)  # noqa
            output_list.append(self.maybe_reconstruct_special(_item))
        return output_list

    def send_set(self, data: set, dst_rank: int, send_type: bool = False):
        """
        Send a set to the destination rank.

        Args:
            data (set): The set to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, set), f"wrong type: {data} must be {set} type."

        if send_type is True:
            self.send_type(set, dst_rank=dst_rank)

        self.send_list(list(data), dst_rank=dst_rank, send_type=False)

    def recv_set(self, src_rank: int) -> set:
        """
        Receive a set from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            set: The received set.
        """
        output_list = self.recv_list(src_rank=src_rank)
        return set(output_list)

    def send_tuple(self, data: tuple, dst_rank: int, send_type: bool = False):
        """
        Send a tuple to the destination rank.

        Args:
            data (tuple): The tuple to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, tuple), f"wrong type: {data} must be {tuple} type."

        if send_type is True:
            self.send_type(tuple, dst_rank=dst_rank)

        self.send_list(list(data), dst_rank=dst_rank, send_type=False)

    def recv_tuple(self, src_rank: int) -> tuple:
        """
        Receive a tuple from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            tuple: The received tuple.
        """
        output_list = self.recv_list(src_rank=src_rank)
        return tuple(output_list)

    def send_size(self, data: torch.Size, dst_rank: int, send_type: bool = False):
        """
        Send a torch.Size to the destination rank.

        Args:
            data (torch.Size): The torch.Size to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, torch.Size), f"wrong type: {data} must be {torch.Size} type."

        if send_type is True:
            self.send_type(torch.Size, dst_rank=dst_rank)

        self.send_list(list(data), dst_rank=dst_rank, send_type=False)

    def recv_size(self, src_rank: int) -> torch.Size:
        """
        Receive a torch.Size from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            torch.Size: The received torch.Size.
        """
        output_list = self.recv_list(src_rank=src_rank)
        return torch.Size(output_list)

    def send_dict(self, data: dict, dst_rank: int, send_type: bool = False):
        """
        Send a dictionary to the destination rank.

        Args:
            data (dict): The dictionary to send.
            dst_rank (int): The destination rank to send the data to.
            send_type (bool): Whether to send the type information.
        """
        assert isinstance(data, dict), f"wrong type: {data} must be {dict} type."

        if send_type is True:
            self.send_type(dict, dst_rank=dst_rank)

        length = len(data)
        self.send_int(length, dst_rank=dst_rank)

        for key, val in data.items():
            key_type, value_type = type(key), type(val)
            assert key_type in self.id_to_dtype, f"unsupported type: {key_type}"
            assert value_type in self.id_to_dtype, f"unsupported type: {value_type}"
            self.instructions[key_type]["send"](key, dst_rank=dst_rank, send_type=True)  # noqa
            self.instructions[value_type]["send"](val, dst_rank=dst_rank, send_type=True)  # noqa

    def recv_dict(self, src_rank: int) -> dict:
        """
        Receive a dictionary from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            dict: The received dictionary.
        """
        output_dict = {}

        length = self.recv_int(src_rank=src_rank)

        for _ in range(length):
            key_type = self.instructions[type]["recv"](src_rank=src_rank)  # noqa
            assert key_type in self.id_to_dtype, f"unsupported type: {key_type}"
            key = self.instructions[key_type]["recv"](src_rank=src_rank)  # noqa

            value_type = self.instructions[type]["recv"](src_rank=src_rank)  # noqa
            assert value_type in self.id_to_dtype, f"unsupported type: {value_type}"
            value = self.instructions[value_type]["recv"](src_rank=src_rank)  # noqa

            key = self.maybe_reconstruct_special(key)
            value = self.maybe_reconstruct_special(value)
            if is_hashable(key):
                output_dict[key] = value
            else:
                raise TypeError(f"Unhashable dict key on receive: " f"{type(key).__name__} -> {key!r}")

        return output_dict

    def send_data_object(self, data: Any, dst_rank: int):
        """
        Send a dataclass object to the destination rank.

        Args:
            data (Any): The dataclass object to send.
            dst_rank (int): The destination rank to send the data to.
        """
        self.send_dict(self.pack_dataclass(data), dst_rank=dst_rank, send_type=True)

    def recv_data_object(self, src_rank: int) -> Any:
        """
        Receive a dataclass object from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            Any: The received dataclass object.
        """
        payload = self.recv_dict(src_rank=src_rank)
        assert (
            isinstance(payload, dict) and payload.get("__kind__") == "dataclass"
        ), "recv_data_class expected a dataclass payload"

        cls = resolve_qualname(payload["module"], payload["qualname"])
        assert is_dataclass(cls), f"received class {cls} is not a dataclass"

        state = payload["state"]

        try:
            init_kwargs = {}
            non_init_items = {}
            for field in fields(cls):
                if field.init:
                    if field.name in state:
                        init_kwargs[field.name] = state[field.name]
                    else:
                        if field.default is not MISSING:
                            init_kwargs[field.name] = field.default
                        elif getattr(field, "default_factory", MISSING) is not MISSING:
                            init_kwargs[field.name] = field.default_factory()
                        else:
                            raise TypeError(f"missing required field: {field.name}")
                else:
                    if field.name in state:
                        non_init_items[field.name] = state[field.name]

            instance = cls(**init_kwargs)
            for key, value in non_init_items.items():
                setattr_frozen_safe(instance, key, value)
            return instance

        except Exception:
            instance = get_object_without_init(cls)
            for key, value in state.items():
                try:
                    setattr_frozen_safe(instance, key, value)
                except Exception:
                    pass
            return instance

    def send_cache(self, data: Any, dst_rank: int):
        """
        Send a huggingface cache object to the destination rank.

        Args:
            data (Any): The huggingface cache object to send.
            dst_rank (int): The destination rank to send the data to.
        """
        self.send_dict(self.pack_cache(data), dst_rank=dst_rank, send_type=True)

    def recv_cache(self, src_rank: int) -> Any:
        """
        Receive a huggingface cache object from the source rank.

        Args:
            src_rank (int): The source rank to receive the data from.

        Returns:
            Any: The received huggingface cache object.
        """
        payload = self.recv_dict(src_rank=src_rank)
        assert isinstance(payload, dict) and payload.get("__kind__") == "hf_cache"
        cls = resolve_qualname(payload["module"], payload["qualname"])
        instance = get_object_without_init(cls)
        for key, value in payload.get("state", {}).items():
            try:
                setattr(instance, key, value)
            except Exception:
                pass
        return instance
