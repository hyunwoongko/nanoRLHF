from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Dict
from collections import Counter

import torch
from torch import nn

from nanorlhf.nanotron.utils.huggingface import is_causal_lm


class ModuleType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    HEAD = "head"
    EMBEDDING = "embedding"
    OTHER = "other"


class AttentionType(Enum):
    QKV_FUSED = 3
    KV_FUSED = 2
    NOT_FUSED = 1


class SlicingType(Enum):
    COLUMN = 0
    ROW = 1
    REPLICATE = None


@dataclass
class ModuleParallelPlan:
    """
    Intra layer tensor parallelization plan for a module.
    """

    module: nn.Module
    module_type: ModuleType
    attention_type: Optional[AttentionType]
    slicing_type: Optional[SlicingType]
    is_reversed: Optional[bool]


@dataclass
class ModelParallelPlan:
    """
    Inter layer pipeline parallelization plan for a model.
    """

    main_module_list: nn.ModuleList
    main_module_list_plans: List[ModuleParallelPlan]
    pre_module_list_plans: List[ModuleParallelPlan]
    post_module_list_plans: List[ModuleParallelPlan]
    embedding_plan: Optional[ModuleParallelPlan]
    head_plan: Optional[ModuleParallelPlan]
    tied_plan: Optional[Tuple[ModuleParallelPlan, ModuleParallelPlan]]

    def extract_modules(self):
        embeddings = [] if self.embedding_plan is not None else []
        pre_modules = [p.module for p in self.pre_module_list_plans]
        post_modules = [p.module for p in self.post_module_list_plans]
        heads = [] if self.head_plan is None else [self.head_plan.module]
        return embeddings, pre_modules, post_modules, heads


class ModelParallelTracer:
    """
    Trace a PyTorch model to generate a model parallelization plan.

    Args:
        model (nn.Module): The PyTorch model to trace.
    """

    def __init__(self, model: nn.Module):
        self.name2module = {}
        self.id2name = {}
        self.id2parent = {}
        self.parent_names = {}
        self.module_lists = {}
        self.module_list_children_ids = {}
        self.call_order_map = {}

        self.model = model
        self.input_embedding = self.get_input_embedding()
        self.output_embedding = self.get_output_embedding()

    def trace(self) -> ModelParallelPlan:
        """
        Trace the model to generate a model parallelization plan.

        Returns:
            ModelParallelPlan: The generated model parallelization plan.
        """
        # 1. build name graph
        self.build_name_graph()

        # 2. create call order map by dummy forward pass
        inputs = self.ensure_dummy_input()
        self.call_order_map = self.record_call_order(inputs)

        # 3. detect main module list
        module_list_name, elements_called = self.detect_main_module_list(min_len=1)
        if module_list_name is None:
            return ModelParallelPlan(
                main_module_list=nn.ModuleList(),
                main_module_list_plans=[],
                pre_module_list_plans=[],
                post_module_list_plans=[],
                embedding_plan=None,
                head_plan=None,
                tied_plan=None,
            )

        # 3. collect module list elements
        module_list = self.module_lists[module_list_name]
        element_ids = [module_id for module_id, _ in elements_called]
        first_index = elements_called[0][1]
        last_index = elements_called[-1][1]
        ids_set = set(element_ids)

        # 4. detect inter layer modules
        outside_called = self.collect_called_outside(ids_set, module_list, id(self.model))
        pre_candidates, post_candidates = self.split_pre_post(outside_called, first_index, last_index)
        embedding, pre_module_list = self.collect_pre_modules(pre_candidates)
        head_by_order, post_module_list = self.collect_post_modules(post_candidates)
        head = self.output_embedding if self.output_embedding is not None else head_by_order
        has_tied = self.has_tied_embedding(head, embedding)

        # 5. build intra layer plans
        embedding_plan = self.build_module_plan(embedding) if embedding is not None else None
        head_plan = self.build_module_plan(head, is_head=True) if head is not None else None
        tied_plan = (embedding_plan, head_plan) if has_tied else None

        pre_module_list_plans = []
        for pre_module in pre_module_list:
            pre_module_plan = self.build_module_plan(pre_module)
            if pre_module_plan is not None:
                pre_module_list_plans.append(pre_module_plan)

        post_module_list_plans = []
        for post_module in post_module_list:
            post_module_plan = self.build_module_plan(post_module)
            if post_module_plan is not None:
                post_module_list_plans.append(post_module_plan)

        module_list_plans = []
        for module in module_list:
            attention_parents = self.collect_attention_parents(module)
            attention_type_map = self.build_attention_child_type_map_for_block(module)
            attention_last_2d_map = self.build_attention_last_2d_map(module)
            for submodule in module.modules():
                if module is submodule:
                    continue
                module_plan = self.build_module_plan(
                    submodule,
                    attention_parents=attention_parents,
                    attention_type_map=attention_type_map,
                    attention_last_2d_map=attention_last_2d_map,
                )
                if module_plan is not None:
                    module_list_plans.append(module_plan)

        return ModelParallelPlan(
            main_module_list=module_list,
            main_module_list_plans=module_list_plans,
            pre_module_list_plans=pre_module_list_plans,
            post_module_list_plans=post_module_list_plans,
            embedding_plan=embedding_plan,
            head_plan=head_plan,
            tied_plan=tied_plan,
        )

    def build_module_plan(
        self,
        module: nn.Module,
        is_head: bool = False,
        attention_parents: Optional[List[nn.Module]] = None,
        attention_type_map: Optional[Dict[int, AttentionType]] = None,
        attention_last_2d_map: Optional[Dict[int, int]] = None,
    ) -> Optional[ModuleParallelPlan]:
        """
        Build a module parallelization plan for a given module.

        Args:
            module (nn.Module): The module to build the plan for.
            is_head (bool): Whether the module is the model head.
            attention_parents (Optional[List[nn.Module]]): List of attention parent modules.
            attention_type_map (Optional[Dict[int, AttentionType]]): Map of attention child types.
            attention_last_2d_map (Optional[Dict[int, int]]): Map of last 2D weights in attention parents.

        Returns:
            Optional[ModuleParallelPlan]:
                The generated module parallelization plan, or `None` if the module has no parameters.
        """
        if not self.has_parameters(module):
            return None

        module_type = self.detect_module_type(module, is_head=is_head)
        is_reversed = self.detect_is_reversed(module, module_type)
        slicing_type = self.detect_slicing_type(
            module, module_type, attention_parents, attention_last_2d_map, is_reversed
        )

        if module_type == ModuleType.ATTENTION:
            attention_type = (attention_type_map or {}).get(id(module))
        else:
            attention_type = None

        return ModuleParallelPlan(
            module=module,
            module_type=module_type,
            attention_type=attention_type,
            slicing_type=slicing_type,
            is_reversed=is_reversed,
        )

    def build_name_graph(self):
        """
        Build mappings between module names, ids, and parent-child relationships.
        """
        for name, module in self.model.named_modules():
            module_id = id(module)
            self.name2module[name] = module
            self.id2name[module_id] = name
            self.parent_names[name] = name.rsplit(".", 1)[0] if "." in name else None

            if isinstance(module, nn.ModuleList):
                self.module_lists[name] = module
                self.module_list_children_ids[module_id] = [id(child) for child in module]

        for name, module in self.name2module.items():
            module_id = id(module)
            parent_name = self.parent_names[name]
            self.id2parent[module_id] = id(self.name2module[parent_name]) if parent_name is not None else None

    def detect_module_type(
        self,
        module: nn.Module,
        is_head: bool,
        attention_parents: Optional[List[nn.Module]] = None,
    ) -> ModuleType:
        """
        Detect the module type for a given module.

        Args:
            module (nn.Module): The module to detect the type for.
            is_head (bool): Whether the module is the model head.
            attention_parents (Optional[List[nn.Module]]): List of attention parent modules.
        """
        if is_head:
            return ModuleType.HEAD

        if isinstance(module, nn.Embedding):
            if self.is_token_embedding(module):
                return ModuleType.EMBEDDING
            else:
                return ModuleType.OTHER

        if self.is_weight_1d_norm(module):
            return ModuleType.NORM

        if attention_parents is not None:
            if self.nearest_attention_parent(module, attention_parents) is not None:
                return ModuleType.ATTENTION
        else:
            if self.has_attention_in_ancestors(module):
                return ModuleType.ATTENTION

        if self.has_2d_weight(module):
            return ModuleType.MLP

        return ModuleType.OTHER

    def detect_attention_type(self, attention_parent: nn.Module) -> Dict[int, AttentionType]:
        """
        Detect the attention type for the children of a given attention parent module.

        Args:
            attention_parent (nn.Module): The attention parent module to analyze.

        Returns:
            Dict[int, AttentionType]: A mapping from child module ids to their detected attention types.
        """
        # TODO: add deepseek-v3 style MLA detection
        parent_name = attention_parent.__class__.__qualname__
        children = list(self.iter_2d_weight_children(attention_parent))
        n = len(children)
        child_ids = [id(m) for (m, _, _) in children]
        result: Dict[int, AttentionType] = {cid: AttentionType.NOT_FUSED for cid in child_ids}

        if n < 2:
            raise ValueError(f"Need at least two projection weights in `{parent_name}` module; got {n}.")
        if n == 2:
            (m0, n0, _), (m1, n1, _) = children
            n0, n1 = int(n0), int(n1)
            if n0 == n1:
                raise ValueError(
                    "Found two projection weights with identical sizes; "
                    "cannot distinguish fused QKV from the output projection "
                    f"in `{parent_name}` module."
                )
            big_idx = 0 if n0 > n1 else 1
            result[id(children[big_idx][0])] = AttentionType.QKV_FUSED
            return result
        if n == 3:
            nums = [int(n) for _, n, _ in children]
            cnt = Counter(nums)
            if len(cnt) == 2:
                unique_numel = next(k for k, v in cnt.items() if v == 1)
                for m, n, _ in children:
                    result[id(m)] = AttentionType.KV_FUSED if int(n) == unique_numel else AttentionType.NOT_FUSED
                return result
            if len(cnt) == 1:
                raise ValueError(
                    "Detected three projection weights with identical sizes; "
                    "cannot distinguish fused KV from query and output projections "
                    f"in `{parent_name}` module."
                )
            raise ValueError(
                "Detected three projection weights with all different sizes; "
                "cannot distinguish fused KV from query and output projections "
                f"in `{parent_name}` module."
            )
        if n == 4:
            return result
        if n > 4:
            raise ValueError(
                f"Detected more than four projection weights ({n}) in `{parent_name}` module; "
                f"cannot automatically determine attention type."
            )
        return result

    def detect_slicing_type(
        self,
        module: nn.Module,
        module_type: ModuleType,
        attention_parents: Optional[List[nn.Module]],
        attention_last_2d_map: Optional[Dict[int, int]],
        is_reversed: bool,
    ):
        """
        Detect the slicing type for a given module.

        Args:
            module (nn.Module): The module to detect the slicing type for.
            module_type (ModuleType): The detected module type.
            attention_parents (Optional[List[nn.Module]]): List of attention parent modules.
            attention_last_2d_map (Optional[Dict[int, int]]): Map of last 2D weights in attention parents.
            is_reversed (bool): Whether the module's weight dimensions are reversed.

        Returns:
            Optional[SlicingType]: The detected slicing type, or `None` if undetermined.
        """
        if module_type == ModuleType.HEAD:
            if is_causal_lm(self.model):
                return SlicingType.COLUMN
            else:
                return SlicingType.REPLICATE

        if isinstance(module, nn.Embedding):
            if self.is_token_embedding(module):
                return SlicingType.COLUMN
            else:
                return SlicingType.REPLICATE

        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return None

        if module_type == ModuleType.NORM or weight.dim() == 1:
            return SlicingType.REPLICATE

        if module_type == ModuleType.ATTENTION:
            if not attention_parents or not attention_last_2d_map:
                raise ValueError("cannot determine attention slicing: missing parents or last-2d map")
            parent = self.nearest_attention_parent(module, attention_parents)
            if parent is None:
                raise ValueError("cannot determine attention slicing: no nearest attention parent")
            last_id = attention_last_2d_map.get(id(parent))
            if last_id is None:
                raise ValueError("cannot determine attention slicing: parent has no last-2d child")
            return SlicingType.ROW if id(module) == last_id else SlicingType.COLUMN

        if module_type == ModuleType.MLP:
            in_dim, out_dim = self.in_out_from_weight(weight, is_reversed)
            return SlicingType.COLUMN if out_dim > in_dim else SlicingType.ROW

        return None

    @staticmethod
    def detect_is_reversed(module: nn.Module, module_type: ModuleType) -> Optional[bool]:
        """
        Detect whether the weight dimensions of a given module are reversed.

        Args:
            module (nn.Module): The module to analyze.
            module_type (ModuleType): The detected module type.

        Returns:
            Optional[bool]: `True` if reversed, `False` if not, or `None` if undetermined.
        """
        if module_type in (ModuleType.EMBEDDING, ModuleType.NORM, ModuleType.OTHER):
            return None
        elif isinstance(module, nn.Embedding):
            return None
        elif isinstance(module, nn.Linear):
            return True
        else:
            return False

    def collect_attention_parents(self, block: nn.Module) -> List[nn.Module]:
        """
        Collect topmost attention parent modules within a given block.

        Args:
            block (nn.Module): The block to analyze.

        Returns:
            List[nn.Module]: A list of topmost attention parent modules.
        """
        candidates: List[nn.Module] = []
        for module in block.modules():
            if self.class_name_has_attention(module):
                candidates.append(module)
        ids = set(id(x) for x in candidates)
        topmost: List[nn.Module] = []
        for module in candidates:
            keep = True
            parent = self.id2parent.get(id(module))
            while parent is not None:
                if parent in ids:
                    keep = False
                    break
                parent = self.id2parent.get(parent)
            if keep:
                topmost.append(module)
        return topmost

    def build_attention_last_2d_map(self, block: nn.Module) -> Dict[int, int]:
        """
        Build a map of the last 2D weight child for each attention parent in a given block.

        Args:
            block (nn.Module): The block to analyze.

        Returns:
            Dict[int, int]: A mapping from attention parent ids to their last 2D weight child ids.
        """
        parents = self.collect_attention_parents(block)
        last_map: Dict[int, int] = {}
        for parent in parents:
            last_by_call_id = None
            last_by_call_seen = -1
            last_by_traversal_id = None

            for child in parent.modules():
                if child is parent:
                    continue
                weight = getattr(child, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.dim() == 2:
                    cid = id(child)
                    seen = self.call_order_map.get(cid, -1)
                    if seen > last_by_call_seen:
                        last_by_call_seen = seen
                        last_by_call_id = cid
                    last_by_traversal_id = cid
            chosen = last_by_call_id if last_by_call_id is not None else last_by_traversal_id
            if chosen is None:
                raise ValueError("cannot determine attention slicing: parent has no 2D child")
            last_map[id(parent)] = chosen
        return last_map

    def build_attention_child_type_map_for_block(self, block: nn.Module) -> Dict[int, AttentionType]:
        """
        Build a map of attention child types for all attention parents in a given block.

        Args:
            block (nn.Module): The block to analyze.

        Returns:
            Dict[int, AttentionType]: A mapping from child module ids to their detected attention types.
        """
        child_map: Dict[int, AttentionType] = {}
        attention_parents = self.collect_attention_parents(block)
        for parent in attention_parents:
            per_parent = self.detect_attention_type(parent)
            child_map.update(per_parent)
        return child_map

    def iter_2d_weight_children(self, parent: nn.Module):
        """
        Iterate over 2D weight children of a given parent module.

        Args:
            parent (nn.Module): The parent module to analyze.

        Yields:
            Tuple[nn.Module, int, int]:
                Yields tuples of (child module, number of elements in weight, call order).
        """
        default_order = 10**12
        for submodule in parent.modules():
            if submodule is parent:
                continue
            weight = getattr(submodule, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dim() == 2:
                numel = int(weight.numel())
                order = self.call_order_map.get(id(submodule), default_order)
                yield submodule, numel, order

    @staticmethod
    def in_out_from_weight(weight: torch.Tensor, is_reversed: Optional[bool]) -> Tuple[int, int]:
        """
        Get the input and output dimensions from a weight tensor.

        Args:
            weight (torch.Tensor): The weight tensor to analyze.
            is_reversed (Optional[bool]): Whether the weight dimensions are reversed.

        Returns:
            Tuple[int, int]: A tuple of (input dimension, output dimension).
        """
        if is_reversed is True:
            out_dim, in_dim = weight.shape[0], weight.shape[1]
        else:
            in_dim, out_dim = weight.shape[0], weight.shape[1]
        return in_dim, out_dim

    @staticmethod
    def is_weight_1d_norm(module: nn.Module) -> bool:
        """
        Check if a module is a 1D weight normalization layer.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if the module is a 1D weight normalization layer, False otherwise.
        """
        weight = getattr(module, "weight", None)
        return isinstance(weight, torch.Tensor) and weight.dim() == 1

    def is_token_embedding(self, module: nn.Module) -> bool:
        """
        Check if a module is the token embedding layer of the model.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if the module is the token embedding layer, False otherwise.
        """
        input_embedding = self.input_embedding
        if input_embedding is None:
            return False
        if module is input_embedding:
            return True
        try:
            return self.have_same_storage(
                getattr(input_embedding, "weight", None),
                getattr(module, "weight", None),
            )
        except Exception:
            return False

    @staticmethod
    def have_same_storage(w1: Optional[torch.Tensor], w2: Optional[torch.Tensor]) -> bool:
        """
        Check if two tensors share the same underlying storage.

        Args:
            w1 (Optional[torch.Tensor]): The first tensor.
            w2 (Optional[torch.Tensor]): The second tensor.

        Returns:
            bool: True if both tensors share the same storage and shape, False otherwise.
        """
        try:
            return (
                isinstance(w1, torch.Tensor)
                and isinstance(w2, torch.Tensor)
                and (w1.untyped_storage().data_ptr() == w2.untyped_storage().data_ptr())
                and (w1.shape == w2.shape)
            )
        except Exception:
            return False

    @staticmethod
    def class_name_has_attention(module: nn.Module) -> bool:
        """
        Check if the class name of a module indicates it is an attention module.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if the module's class name contains "attention" or "attn", False otherwise.
        """
        name = module.__class__.__qualname__.lower()
        return ("attention" in name) or ("attn" in name)

    def has_attention_in_ancestors(self, module: nn.Module) -> bool:
        """
        Check if any ancestor of a module is an attention module.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if any ancestor is an attention module, False otherwise.
        """
        current = self.id2parent.get(id(module))
        while current is not None:
            parent_module = self.name2module[self.id2name[current]]
            if self.class_name_has_attention(parent_module):
                return True
            current = self.id2parent.get(current)
        return False

    def has_2d_weight(self, module: nn.Module) -> bool:
        """
        Check if a module has any 2D weight parameters.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if the module has any 2D weight parameters, False otherwise.
        """
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            return True
        for parameter in module.parameters(recurse=False):
            if isinstance(parameter, torch.Tensor) and parameter.dim() == 2:
                return True
        return False

    def has_tied_embedding(self, embedding: Optional[nn.Module], head: Optional[nn.Module]) -> bool:
        """
        Check if the embedding and head modules share tied weights.

        Args:
            embedding (Optional[nn.Module]): The embedding module.
            head (Optional[nn.Module]): The head module.

        Returns:
            bool: True if the embedding and head share tied weights, False otherwise.
        """
        if embedding is not None and head is not None:
            if self.have_same_storage(getattr(embedding, "weight", None), getattr(head, "weight", None)):
                return True
        return False

    def has_parameters(self, module: nn.Module) -> bool:
        """
        Check if a module has any parameters.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if the module has parameters, False otherwise.
        """
        for _ in module.parameters(recurse=False):
            return True
        return False

    def nearest_attention_parent(self, module: nn.Module, attention_parents: List[nn.Module]) -> Optional[nn.Module]:
        """
        Find the nearest attention parent module for a given module.

        Args:
            module (nn.Module): The module to analyze.
            attention_parents (List[nn.Module]): List of attention parent modules.

        Returns:
            Optional[nn.Module]: The nearest attention parent module, or `None` if not found.
        """
        current = self.id2parent.get(id(module), None)
        attention_ids = {id(a) for a in attention_parents}
        while current is not None:
            if current in attention_ids:
                return self.name2module[self.id2name[current]]
            current = self.id2parent.get(current)
        return None

    def ensure_dummy_input(self) -> torch.Tensor:
        """
        Ensure a dummy input tensor for the model.

        Returns:
            torch.Tensor: A dummy input tensor.
        """
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.randint(0, 10, (1, 4), dtype=torch.long, device=device)

    def record_call_order(self, inputs: torch.Tensor) -> Dict[int, int]:
        """
        Record the call order of modules during a forward pass.

        Args:
            inputs (torch.Tensor): The input tensor for the model.
        """
        first_seen: Dict[int, int] = {}
        counter = {"i": 0}
        handles = []

        def pre_hook(module: nn.Module, _):
            module_id = id(module)
            if module_id not in first_seen:
                first_seen[module_id] = counter["i"]
            counter["i"] += 1

        for module in self.model.modules():
            handles.append(module.register_forward_pre_hook(pre_hook))

        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs)

        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

        return first_seen

    def detect_main_module_list(self, min_len: int) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        """
        Detect the main module list based on call order.

        Args:
            min_len (int): Minimum length of the module list to consider.
        """
        best_name: Optional[str] = None
        best_ordered: List[Tuple[int, int]] = []
        best_count = -1

        for list_name, module_list in self.module_lists.items():
            element_ids = self.module_list_children_ids[id(module_list)]
            called = [
                (module_id, self.call_order_map[module_id])
                for module_id in element_ids
                if module_id in self.call_order_map
            ]
            if not called:
                continue
            called.sort(key=lambda t: t[1])
            if len(called) > best_count and len(called) >= min_len:
                best_count = len(called)
                best_name = list_name
                best_ordered = called
        return best_name, best_ordered

    def collect_called_outside(
        self, module_ids_set: set[int], module_list: nn.Module, root_id: int
    ) -> List[Tuple[int, int]]:
        """
        Collect modules called outside the main module list.

        Args:
            module_ids_set (set[int]): Set of module ids in the main module list.
            module_list (nn.Module): The main module list.
            root_id (int): The id of the root model module.

        Returns:
            List[Tuple[int, int]]: List of tuples of (module id, call order index).
        """
        ancestors = self.collect_ancestors(list(module_ids_set))
        ancestors.add(id(module_list))
        ancestors.add(root_id)

        result: List[Tuple[int, int]] = []
        for module_id, index in self.call_order_map.items():
            if module_id == root_id:
                continue
            if self.is_descendant(module_id, module_ids_set):
                continue
            if module_id in ancestors:
                continue
            module_name = self.id2name.get(module_id, "")
            if module_name == "":
                continue
            result.append((module_id, index))
        return result

    def split_pre_post(
        self,
        outside: List[Tuple[int, int]],
        first_index: int,
        last_index: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Split outside called modules into pre and post lists.

        Args:
            outside (List[Tuple[int, int]]): List of tuples of (module id, call order index).
            first_index (int): The first index of the main module list.
            last_index (int): The last index of the main module list.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
                Two lists of tuples of (module id, call order index) for pre and post modules.
        """
        topmost_ids = self.topmost([module_id for module_id, _ in outside])
        topmost_with_index = [(module_id, self.call_order_map[module_id]) for module_id in topmost_ids]
        topmost_with_index.sort(key=lambda t: t[1])

        pre: List[Tuple[int, int]] = []
        post: List[Tuple[int, int]] = []
        for module_id, index in topmost_with_index:
            if index < first_index:
                pre.append((module_id, index))
            elif index > last_index:
                post.append((module_id, index))
        return pre, post

    def collect_pre_modules(
        self, pre_candidates: List[Tuple[int, int]]
    ) -> Tuple[Optional[nn.Module], List[nn.Module]]:
        """
        Collect embedding and pre modules from pre candidates.

        Args:
            pre_candidates (List[Tuple[int, int]]): List of tuples of (module id, call order index).

        Returns:
            Tuple[Optional[nn.Module], List[nn.Module]]:
                The embedding module (if any) and a list of pre modules.
        """
        embedding: Optional[nn.Module] = None
        pre_module_list: List[nn.Module] = []
        for module_id, _ in pre_candidates:
            module = self.name2module[self.id2name[module_id]]
            if self.is_token_embedding(module):
                embedding = module
            else:
                pre_module_list.append(module)
        return embedding, pre_module_list

    def collect_post_modules(
        self, post_candidates: List[Tuple[int, int]]
    ) -> Tuple[Optional[nn.Module], List[nn.Module]]:
        """
        Collect head and post modules from post candidates.

        Args:
            post_candidates (List[Tuple[int, int]]): List of tuples of (module id, call order index).

        Returns:
            Tuple[Optional[nn.Module], List[nn.Module]]:
                The head module (if any) and a list of post modules.
        """
        head_by_order: Optional[nn.Module] = None
        post_module_list: List[nn.Module] = []
        if post_candidates:
            post_candidates.sort(key=lambda t: t[1])
            last_module_id, _ = post_candidates[-1]
            last_module = self.name2module[self.id2name[last_module_id]]
            if isinstance(last_module, nn.Linear):
                head_by_order = last_module
                for module_id, _ in post_candidates[:-1]:
                    post_module_list.append(self.name2module[self.id2name[module_id]])
            else:
                for module_id, _ in post_candidates:
                    post_module_list.append(self.name2module[self.id2name[module_id]])
        return head_by_order, post_module_list

    def collect_ancestors(self, module_ids: List[int]) -> set[int]:
        """
        Collect all ancestor module ids for a list of module ids.

        Args:
            module_ids (List[int]): List of module ids to analyze.

        Returns:
            set[int]: A set of ancestor module ids.
        """
        ancestors: set[int] = set()
        for module_id in module_ids:
            parent = self.id2parent.get(module_id)
            while parent is not None:
                if parent in ancestors:
                    parent = self.id2parent.get(parent)
                    continue
                ancestors.add(parent)
                parent = self.id2parent.get(parent)
        return ancestors

    def is_descendant(self, module_id: int, ancestor_ids: set[int]) -> bool:
        """
        Check if a module is a descendant of any module in a set of ancestor ids.

        Args:
            module_id (int): The module id to check.
            ancestor_ids (set[int]): Set of ancestor module ids.

        Returns:
            bool: True if the module is a descendant of any ancestor, False otherwise.
        """
        current = module_id
        while current is not None:
            if current in ancestor_ids:
                return True
            current = self.id2parent.get(current)
        return False

    def topmost(self, candidates: List[int]) -> List[int]:
        """
        Get the topmost modules from a list of candidate module ids.

        Args:
            candidates (List[int]): List of candidate module ids.

        Returns:
            List[int]: List of topmost module ids.
        """
        candidate_set = set(candidates)
        topmost: List[int] = []
        for module_id in candidates:
            parent = self.id2parent.get(module_id)
            keep = True
            while parent is not None:
                if parent in candidate_set:
                    keep = False
                    break
                parent = self.id2parent.get(parent)
            if keep:
                topmost.append(module_id)
        return topmost

    def get_input_embedding(self) -> Optional[nn.Module]:
        """
        Get the input embedding module of the model.

        Returns:
            Optional[nn.Module]: The input embedding module, or `None` if not found.
        """
        try:
            if hasattr(self.model, "get_input_embeddings"):
                return self.model.get_input_embeddings()
        except Exception:
            pass
        return None

    def get_output_embedding(self) -> Optional[nn.Module]:
        """
        Get the output embedding module of the model.

        Returns:
            Optional[nn.Module]: The output embedding module, or `None` if not found.
        """
        try:
            if hasattr(self.model, "get_output_embeddings"):
                return self.model.get_output_embeddings()
        except Exception:
            pass
        return None
