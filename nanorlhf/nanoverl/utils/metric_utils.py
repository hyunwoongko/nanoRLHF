from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist


@dataclass
class WeightedStat:
    value_sum: torch.Tensor
    weight_sum: torch.Tensor
    value_sq_sum: Optional[torch.Tensor]


@dataclass
class ScalarStat:
    value_sum: torch.Tensor
    count: torch.Tensor
    value_sq_sum: Optional[torch.Tensor]


@dataclass
class ReducedStat:
    value: torch.Tensor
    reduce_op: dist.ReduceOp
    nonfinite_value: float


class MetricsAccumulator:
    """
    Accumulates various types of metrics during training or evaluation.
    This is using in RLTrainer to collect and aggregate metrics across micro-batches

    Args:
        device (torch.device): The device to store the accumulated metrics.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.sum_stats: Dict[str, torch.Tensor] = {}
        self.scalar_stats: Dict[str, ScalarStat] = {}
        self.weighted_stats: Dict[str, WeightedStat] = {}
        self.reduced_min_stats: Dict[str, ReducedStat] = {}
        self.reduced_max_stats: Dict[str, ReducedStat] = {}

    def reset(self):
        """
        Resets all accumulated metrics.
        """
        self.sum_stats.clear()
        self.weighted_stats.clear()
        self.scalar_stats.clear()
        self.reduced_min_stats.clear()
        self.reduced_max_stats.clear()

    def add_sum(self, name: str, value: torch.Tensor):
        """
        Adds a value to a simple sum metric.

        Args:
            name (str): The name of the metric.
            value (torch.Tensor): The value to add.
        """
        if name not in self.sum_stats:
            self.sum_stats[name] = torch.zeros((), device=self.device, dtype=torch.float32)
        self.sum_stats[name] += value.detach().to(device=self.device, dtype=torch.float32)

    def add_many_sums(self, **named_values: torch.Tensor):
        """
        Adds multiple values to their corresponding sum metrics.

        Args:
            **named_values: Named values to add to their respective sum metrics.
        """
        for name, value in named_values.items():
            self.add_sum(name, value)

    def add_weighted_sum(
        self,
        name: str,
        value_sum: torch.Tensor,
        weight_sum: torch.Tensor,
        value_sq_sum: Optional[torch.Tensor] = None,
    ):
        """
        Adds a weighted sum metric.

        Args:
            name (str): The name of the metric.
            value_sum (torch.Tensor): The sum of values to add.
            weight_sum (torch.Tensor): The sum of weights to add.
            value_sq_sum (Optional[torch.Tensor]): The sum of squared values to add.
        """
        if name not in self.weighted_stats:
            self.weighted_stats[name] = WeightedStat(
                value_sum=torch.zeros((), device=self.device, dtype=torch.float32),
                weight_sum=torch.zeros((), device=self.device, dtype=torch.float32),
                value_sq_sum=(
                    torch.zeros((), device=self.device, dtype=torch.float32) if value_sq_sum is not None else None
                ),
            )

        stat = self.weighted_stats[name]
        stat.value_sum += value_sum.detach().to(device=self.device, dtype=torch.float32)
        stat.weight_sum += weight_sum.detach().to(device=self.device, dtype=torch.float32)
        if value_sq_sum is not None:
            if stat.value_sq_sum is None:
                stat.value_sq_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            stat.value_sq_sum += value_sq_sum.detach().to(device=self.device, dtype=torch.float32)

    def add_many_weighted_sums(self, **named_triplets):
        """
        Adds multiple weighted sum metrics.

        Args:
            **named_triplets: Named triplets of (value_sum, weight_sum) or (value_sum, weight_sum, value_sq_sum).
        """
        for name, payload in named_triplets.items():
            if isinstance(payload, tuple) and (len(payload) == 2 or len(payload) == 3):
                value_sum = payload[0]
                weight_sum = payload[1]
                value_sq_sum = payload[2] if len(payload) == 3 else None
                self.add_weighted_sum(name, value_sum=value_sum, weight_sum=weight_sum, value_sq_sum=value_sq_sum)
            else:
                raise ValueError(
                    f"Invalid payload for {name}: expected (value_sum, weight_sum) or (value_sum, weight_sum, value_sq_sum)"
                )

    def add_scalar(self, name: str, value: torch.Tensor, with_sq: bool = False):
        """
        Adds a scalar metric.

        Args:
            name (str): The name of the metric.
            value (torch.Tensor): The scalar value to add.
            with_sq (bool): Whether to also accumulate the square of the value.
        """
        if name not in self.scalar_stats:
            self.scalar_stats[name] = ScalarStat(
                value_sum=torch.zeros((), device=self.device, dtype=torch.float32),
                count=torch.zeros((), device=self.device, dtype=torch.float32),
                value_sq_sum=(torch.zeros((), device=self.device, dtype=torch.float32) if with_sq else None),
            )
        stat = self.scalar_stats[name]
        v = value.detach().to(device=self.device, dtype=torch.float32)
        stat.value_sum += v
        stat.count += 1.0
        if with_sq:
            if stat.value_sq_sum is None:
                stat.value_sq_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            stat.value_sq_sum += v * v

    def add_many_scalars(self, with_sq: bool = False, **named_values: torch.Tensor):
        """
        Adds multiple scalar metrics.

        Args:
            with_sq (bool): Whether to also accumulate the square of the values.
            **named_values: Named scalar values to add.
        """
        for name, value in named_values.items():
            self.add_scalar(name, value=value, with_sq=with_sq)

    def update_min(self, name: str, value: torch.Tensor, nonfinite_value: float = 0.0):
        """
        Updates a minimum reduced metric.

        Args:
            name (str): The name of the metric.
            value (torch.Tensor): The value to compare for minimum.
            nonfinite_value (float): The value to use if the input is non-finite.
        """
        if name not in self.reduced_min_stats:
            self.reduced_min_stats[name] = ReducedStat(
                value=torch.full((), float("inf"), device=self.device, dtype=torch.float32),
                reduce_op=dist.ReduceOp.MIN,
                nonfinite_value=float(nonfinite_value),
            )

        stat = self.reduced_min_stats[name]
        stat.nonfinite_value = float(nonfinite_value) if nonfinite_value is not None else stat.nonfinite_value

        v = value.detach().to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(v).item():
            v = torch.tensor(stat.nonfinite_value, device=self.device, dtype=torch.float32)
        stat.value = torch.minimum(stat.value, v)

    def update_max(self, name: str, value: torch.Tensor, nonfinite_value: float = 0.0):
        """
        Updates a maximum reduced metric.

        Args:
            name (str): The name of the metric.
            value (torch.Tensor): The value to compare for maximum.
            nonfinite_value (float): The value to use if the input is non-finite.
        """
        if name not in self.reduced_max_stats:
            self.reduced_max_stats[name] = ReducedStat(
                value=torch.full((), -float("inf"), device=self.device, dtype=torch.float32),
                reduce_op=dist.ReduceOp.MAX,
                nonfinite_value=float(nonfinite_value),
            )

        stat = self.reduced_max_stats[name]
        stat.nonfinite_value = float(nonfinite_value) if nonfinite_value is not None else stat.nonfinite_value

        v = value.detach().to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(v).item():
            v = torch.tensor(stat.nonfinite_value, device=self.device, dtype=torch.float32)
        stat.value = torch.maximum(stat.value, v)

    def update_many_min(self, nonfinite_value: float = 0.0, **named_values: torch.Tensor):
        """
        Updates multiple minimum reduced metrics.

        Args:
            nonfinite_value (float): The value to use if the input is non-finite.
        """
        for name, value in named_values.items():
            self.update_min(name, value=value, nonfinite_value=nonfinite_value)

    def update_many_max(self, nonfinite_value: float = 0.0, **named_values: torch.Tensor):
        """
        Updates multiple maximum reduced metrics.

        Args:
            nonfinite_value (float): The value to use if the input is non-finite.
        """
        for name, value in named_values.items():
            self.update_max(name, value=value, nonfinite_value=nonfinite_value)

    def all_reduce(self, group=None):
        """
        Performs all-reduce operations across multiple processes for distributed training.

        Args:
            group: The process group to use for the all-reduce operation.
        """
        if (not dist.is_available()) or (not dist.is_initialized()):
            return
        if dist.get_world_size(group=group) <= 1:
            return

        for tensor in self.sum_stats.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        for stat in self.weighted_stats.values():
            dist.all_reduce(stat.value_sum, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(stat.weight_sum, op=dist.ReduceOp.SUM, group=group)
            if stat.value_sq_sum is not None:
                dist.all_reduce(stat.value_sq_sum, op=dist.ReduceOp.SUM, group=group)

        for stat in self.scalar_stats.values():
            dist.all_reduce(stat.value_sum, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(stat.count, op=dist.ReduceOp.SUM, group=group)
            if stat.value_sq_sum is not None:
                dist.all_reduce(stat.value_sq_sum, op=dist.ReduceOp.SUM, group=group)

        for stat in self.reduced_min_stats.values():
            dist.all_reduce(stat.value, op=dist.ReduceOp.MIN, group=group)

        for stat in self.reduced_max_stats.values():
            dist.all_reduce(stat.value, op=dist.ReduceOp.MAX, group=group)

    def to_dict(self) -> Dict[str, float]:
        """
        Converts the accumulated metrics to a dictionary.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        output: Dict[str, float] = {}

        for name, value_sum in self.sum_stats.items():
            output[name] = float(value_sum.item())

        for name, stat in self.weighted_stats.items():
            weight = stat.weight_sum.clamp_min(1.0)
            mean = stat.value_sum / weight
            output[name] = float(mean.item())
            if stat.value_sq_sum is not None:
                second = stat.value_sq_sum / weight
                var = torch.clamp(second - mean * mean, min=0.0)
                std = torch.sqrt(var)
                output[f"{name}_std"] = float(std.item())

        for name, stat in self.scalar_stats.items():
            count = stat.count.clamp_min(1.0)
            mean = stat.value_sum / count
            output[name] = float(mean.item())
            if stat.value_sq_sum is not None:
                second = stat.value_sq_sum / count
                var = torch.clamp(second - mean * mean, min=0.0)
                std = torch.sqrt(var)
                output[f"{name}_std"] = float(std.item())

        for name, stat in self.reduced_min_stats.items():
            v = stat.value
            if not torch.isfinite(v).item():
                v = torch.tensor(stat.nonfinite_value, device=self.device, dtype=torch.float32)
            output[name] = float(v.item())

        for name, stat in self.reduced_max_stats.items():
            v = stat.value
            if not torch.isfinite(v).item():
                v = torch.tensor(stat.nonfinite_value, device=self.device, dtype=torch.float32)
            output[name] = float(v.item())

        return output


def accumulate_ppo_micro_metrics(
    metrics,
    *,
    total_loss: torch.Tensor,
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    ratio: torch.Tensor,
    ratio_clipped: torch.Tensor,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    micro_loss_mask: torch.Tensor,
    micro_valid_tokens: torch.Tensor,
    compute_kl_per_token_fn,
):
    """
    Accumulates PPO-related metrics for a micro-batch.

    Args:
        metrics: The MetricsAccumulator instance to accumulate metrics into.
        total_loss (torch.Tensor): The total loss for the micro-batch.
        policy_loss (torch.Tensor): The policy loss for the micro-batch.
        value_loss (torch.Tensor): The value loss for the micro-batch.
        ratio (torch.Tensor): The probability ratio for the micro-batch.
        ratio_clipped (torch.Tensor): The clipped probability ratio for the micro-batch.
        new_logprobs (torch.Tensor): The new log probabilities for the micro-batch.
        ref_logprobs (torch.Tensor): The reference log probabilities for the micro-batch.
        new_values (torch.Tensor): The new value predictions for the micro-batch.
        returns (torch.Tensor): The return targets for the micro-batch.
        advantages (torch.Tensor): The advantage estimates for the micro-batch.
        micro_loss_mask (torch.Tensor): The mask indicating valid tokens for loss computation.
        micro_valid_tokens (torch.Tensor): The number of valid tokens in the micro-batch.
        compute_kl_per_token_fn: Function to compute KL divergence per token.
    """
    mask = micro_loss_mask
    ratio_values = ratio[mask]
    clipped_ratio_values = ratio_clipped[mask]
    clipfrac_values = (clipped_ratio_values != ratio_values).float()
    ppo_kl_values = compute_kl_per_token_fn(new_logprobs, ref_logprobs)[mask]
    value_predictions = new_values[mask]
    return_targets = returns[mask]
    advantage_values = advantages[mask]
    value_residual_sq = (value_predictions - return_targets) ** 2
    negative_log_probability = (-new_logprobs[mask]).float()

    metrics.add_many_sums(
        valid_tokens_sum=micro_valid_tokens,
        returns_sum=return_targets.sum(),
        returns_sq_sum=(return_targets * return_targets).sum(),
        resid2_sum=value_residual_sq.sum(),
    )

    metrics.update_many_min(nonfinite_value=1.0, ratio_min=ratio_values.min())
    metrics.update_many_max(nonfinite_value=1.0, ratio_max=ratio_values.max())
    metrics.update_many_max(nonfinite_value=0.0, ppo_kl_max=ppo_kl_values.max())

    metrics.add_many_weighted_sums(
        loss_total=(total_loss.detach() * micro_valid_tokens, micro_valid_tokens),
        loss_policy=(policy_loss.detach() * micro_valid_tokens, micro_valid_tokens),
        loss_value=(value_loss.detach() * micro_valid_tokens, micro_valid_tokens),
        ppo_kl=(ppo_kl_values.sum(), micro_valid_tokens),
        clipfrac=(clipfrac_values.sum(), micro_valid_tokens),
        entropy_proxy=(negative_log_probability.sum(), micro_valid_tokens),
        ratio=(ratio_values.sum(), micro_valid_tokens),
        vpreds=(value_predictions.sum(), micro_valid_tokens),
        returns=(return_targets.sum(), micro_valid_tokens),
        advantages=(advantage_values.sum(), micro_valid_tokens),
        resid2=(value_residual_sq.sum(), micro_valid_tokens),
    )


def compute_explained_variance(output: dict) -> float:
    """
    Computes the explained variance metric from accumulated sums.

    Args:
        output (dict): A dictionary containing accumulated sums:
            - "valid_tokens_sum": Total number of valid tokens.
            - "returns_sum": Sum of returns.
            - "returns_sq_sum": Sum of squared returns.
            - "resid2_sum": Sum of squared residuals.

    Returns:
        float: The explained variance value.
    """
    valid_tokens_sum = float(output.pop("valid_tokens_sum"))
    returns_sum = float(output.pop("returns_sum"))
    returns_sq_sum = float(output.pop("returns_sq_sum"))
    resid2_sum = float(output.pop("resid2_sum"))

    if valid_tokens_sum <= 0.0:
        return 0.0

    returns_mean = returns_sum / valid_tokens_sum
    returns_second = returns_sq_sum / valid_tokens_sum
    returns_var = returns_second - (returns_mean * returns_mean)
    if returns_var < 0.0:
        returns_var = 0.0

    resid2_mean = resid2_sum / valid_tokens_sum
    denom = max(returns_var, 1e-12)
    explained_variance = 1.0 - (resid2_mean / denom)
    return float(explained_variance)
