from typing import List, Optional


class SchedulingPolicy:
    """
    Abstract interface for node selection among candidate nodes.

    Discussion:
        Q. Why does the policy get a *stable* node order?
            For reproducibility and teaching clarity. FIFO uses the given order;
            RoundRobin advances a cursor over this order while skipping ineligible nodes.
    """

    def set_node_order(self, node_ids: List[str]) -> None:
        """Called by the scheduler to set the global, stable node order."""
        raise NotImplementedError

    def select(self, candidates: List[str]) -> Optional[str]:
        """Choose one node id from `candidates` or return None if none is acceptable."""
        raise NotImplementedError


class FIFO(SchedulingPolicy):
    """
    First-In-First-Out (FIFO) scheduling policy.

    Attributes:
        order (List[str]): The global, stable node order set by the scheduler.

    Examples:
        >>> policy = FIFO()
        >>> policy.set_node_order(['node-A', 'node-B', 'node-C'])
        >>> policy.select(['node-B', 'node-C'])
        'node-B'
    """

    order: List[str] = []

    def set_node_order(self, node_ids: List[str]) -> None:
        """Called by the scheduler to set the global, stable node order."""
        self.order = node_ids.copy()

    def select(self, candidates: List[str]) -> Optional[str]:
        """Choose one node id from `candidates` or return None if none is acceptable."""
        if not candidates or not self.order:
            return None

        cand = set(candidates)
        for nid in self.order:
            if nid in cand:
                return nid
        return None


class RoundRobin(SchedulingPolicy):
    """
    Cycle through the node order and pick the next available candidate.

    Attributes:
        order (List[str]): The global, stable node order set by the scheduler.
        cursor (int): The current position in the node order for round-robin selection.

    Examples:
        >>> policy = RoundRobin()
        >>> policy.set_node_order(['node-A', 'node-B', 'node-C'])
        >>> policy.select(['node-B', 'node-C'])
        'node-B'
        >>> policy.select(['node-B', 'node-C'])
        'node-C'
        >>> policy.select(['node-B', 'node-C'])
        'node-B'
        >>> policy.select(['node-A'])
        'node-A'
        >>> policy.select(['node-D'])
        None

    Discussion:
        Q. What if the next node in round-robin is not eligible?
            The policy scans forward (with wrap-around) until it finds an eligible node.
            If none are eligible, it returns None.
    """
    order: List[str] = []
    cursor: int = 0

    def set_node_order(self, node_ids: List[str]) -> None:
        """Called by the scheduler to set the global, stable node order."""
        self.order = node_ids.copy()
        self.cursor = 0

    def select(self, candidates: List[str]) -> Optional[str]:
        """Choose one node id from `candidates` or return None if none is acceptable."""
        if not candidates or not self.order:
            return None

        candidate_set = set(candidates)
        n = len(self.order)

        for step in range(n):
            idx = (self.cursor + step) % n
            nid = self.order[idx]
            if nid in candidate_set:
                if len(candidates) > 1:
                    # Advance cursor only if there are multiple candidates
                    self.cursor = (idx + 1) % n
                return nid
        return candidates[0]
