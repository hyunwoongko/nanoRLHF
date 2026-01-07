from dataclasses import dataclass, field
from typing import Dict, Optional

from nanorlhf.nanoray.core.placement import Bundle
from nanorlhf.nanoray.core.task import Task


@dataclass
class NodeState:
    total_cpus: float = 1.0
    total_gpus: float = 0.0
    total_custom: Dict[str, float] = field(default_factory=dict)

    used_cpus: float = 0.0
    used_gpus: float = 0.0
    used_custom: Dict[str, float] = field(default_factory=dict)

    reserved_cpus: float = 0.0
    reserved_gpus: float = 0.0
    reserved_custom: Dict[str, float] = field(default_factory=dict)

    def _fit_custom_with_base(self, base: Dict[str, float], req: Optional[Dict[str, float]]) -> bool:
        """
        Check if custom resources in `req` can fit within `total_custom`
        given the current usage in `base`.

        Args:
            base (Dict[str, float]): Current usage of custom resources.
            req (Optional[Dict[str, float]]): Required custom resources to check.
            
        Returns:
            bool: True if requirements can be met, False otherwise.
        """
        if not req:
            return True
        for kind, requirement in req.items():
            total = float(self.total_custom.get(kind, 0.0))
            used = float(base.get(kind, 0.0))
            if used + float(requirement) > total:
                return False
        return True

    def can_run_req(self, cpus: float, gpus: float, resources: Optional[Dict[str, float]]) -> bool:
        """
        Check if the node can run a task with the given resource requirements.
        
        Args:
            cpus (float): Number of CPUs required.
            gpus (float): Number of GPUs required.
            resources (Optional[Dict[str, float]]): Custom resource requirements.
            
        Returns:
            bool: True if the task can be run, False otherwise.
        """
        if self.used_cpus + self.reserved_cpus + float(cpus) > self.total_cpus:
            return False
        if self.used_gpus + self.reserved_gpus + float(gpus) > self.total_gpus:
            return False

        merged_base = dict(self.used_custom)
        for k, v in self.reserved_custom.items():
            merged_base[k] = float(merged_base.get(k, 0.0)) + float(v)

        if not self._fit_custom_with_base(merged_base, resources):
            return False
        return True

    def can_run(self, task: Task) -> bool:
        """
        Check if the node can run the given task based on its resource requirements.
        
        Args:
            task (Task): The task to check.
            
        Returns:
            bool: True if the task can be run, False otherwise.
        """
        return self.can_run_req(task.num_cpus, task.num_gpus, task.resources)

    def can_reserve_bundle(self, bundle: Bundle) -> bool:
        """
        Check if the node can reserve resources for the given bundle.
        
        Args:
            bundle (Bundle): The resource bundle to check.
        
        Returns:
            bool: True if the bundle can be reserved, False otherwise.
        """
        return self.can_run_req(bundle.cpus, bundle.gpus, bundle.resources)

    def reserve_bundle(self, bundle: Bundle):
        """
        Reserve resources for the given bundle.
        
        Args:
            bundle (Bundle): The resource bundle to reserve.
        """
        self.reserved_cpus += float(bundle.cpus)
        self.reserved_gpus += float(bundle.gpus)
        if bundle.resources:
            for k, v in bundle.resources.items():
                self.reserved_custom[k] = float(self.reserved_custom.get(k, 0.0)) + float(v)

    def release_bundle(self, bundle: Bundle):
        """
        Release reserved resources for the given bundle.

        Args:
            bundle (Bundle): The resource bundle to release.
        """
        self.reserved_cpus -= float(bundle.cpus)
        self.reserved_gpus -= float(bundle.gpus)
        if self.reserved_cpus < 0.0:
            self.reserved_cpus = 0.0
        if self.reserved_gpus < 0.0:
            self.reserved_gpus = 0.0

        if bundle.resources:
            for k, v in bundle.resources.items():
                new_v = float(self.reserved_custom.get(k, 0.0)) - float(v)
                if new_v <= 0.0:
                    self.reserved_custom.pop(k, None)
                else:
                    self.reserved_custom[k] = new_v

    def allocate(self, task: Task):
        """
        Allocate resources for the given task.

        Args:
            task (Task): The task for which to allocate resources.
        """
        self.used_cpus += task.num_cpus
        self.used_gpus += task.num_gpus
        if task.resources:
            for k, v in task.resources.items():
                self.used_custom[k] = float(self.used_custom.get(k, 0.0)) + float(v)

    def release(self, task: Task):
        """
        Release allocated resources for the given task.

        Args:
            task (Task): The task for which to release resources.
        """
        self.used_cpus -= task.num_cpus
        self.used_gpus -= task.num_gpus
        if self.used_cpus < 0.0:
            self.used_cpus = 0.0
        if self.used_gpus < 0.0:
            self.used_gpus = 0.0

        if task.resources:
            for k, v in task.resources.items():
                new_v = float(self.used_custom.get(k, 0.0)) - float(v)
                if new_v <= 0.0:
                    self.used_custom.pop(k, None)
                else:
                    self.used_custom[k] = new_v
