from nanorlhf.nanoray.core.object_ref import ObjectRef
from nanorlhf.nanoray.core.task import Task
from nanorlhf.nanoray.network.rpc_client import RpcClient


class RemoteWorkerProxy:
    """
    Worker-like proxy that forwards `execute_task` to a remote node via RPC.

    This class *duck types* the `WorkerLike` protocol used by the Scheduler.
    No need to import the protocol to avoid cycles.

    Args:
        node_id (str): The remote node id to execute tasks on.
        rpc (RpcClient): Configured RPC client.

    Examples:
        >>> # sched = Scheduler(policy, nodes={"A": (Worker(...), cap), "B": (RemoteWorkerProxy("B", rpc), cap)})
    """

    def __init__(self, node_id: str, rpc: RpcClient):
        self.node_id = node_id
        self.rpc = rpc

    def execute_task(self, task: Task) -> ObjectRef:
        """
        Execute a task remotely via RPC.

        Args:
            task (Task): The task to execute.
        """
        return self.rpc.execute_task(self.node_id, task)
