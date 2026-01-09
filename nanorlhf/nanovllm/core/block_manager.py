from collections import deque

import numpy as np
import xxhash


class Block:
    """
    A block is a fixed-size chunk of token IDs.

    Attributes:
        block_id (int): The unique identifier for the block.
        ref_count (int): The number of sequences currently using this block.
        hash (int): The hash value of the block's token IDs.
        token_ids (list): The list of token IDs in the block.
    """
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        # ref_count=0 means that no sequence is using this block
        self.hash = -1
        self.token_ids = []

    def update(self, hash, token_ids):
        """
        Update the block's hash and token IDs.

        Args:
            hash (int): The new hash value for the block.
            token_ids (list): The new list of token IDs for the block.
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        Reset the block's attributes to default values.
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    BlockManager manages a pool of blocks for sequences.

    Attributes:
        block_size (int): The size of each block.
        blocks (list): The list of Block objects.
        hash_to_block_id (dict): A mapping from block hash to block ID.
        free_block_ids (deque): A deque of free block IDs.
        used_block_ids (set): A set of used block IDs.
    """
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    @classmethod
    def compute_hash(cls, token_ids, prefix=-1):
        """
        Compute the hash value for a block of token IDs.

        Args:
            token_ids (list): The list of token IDs in the block.
            prefix (int): The hash of the previous block, or -1 if none.

        Returns:
            int: The computed hash value.
        """
        # (previous blocks + current block) is hashed for uniqueness.
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, 'little'))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def allocate_block(self, block_id):
        """
        Allocate a block by block_id.

        Args:
            block_id (int): The ID of the block to allocate.

        Returns:
            Block: The allocated Block object.
        """
        # get a block by block_id from free_block_ids, move it to used_block_ids,
        # and set its ref_count to 1 (this means it is being used by one sequence)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def deallocate_block(self, block_id):
        """
        Deallocate a block by block_id.

        Args:
            block_id (int): The ID of the block to deallocate.
        """
        # release a block by block_id which is no longer used by any sequence,
        # so we remove it from used_block_ids and add it back to free_block_ids again.
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, sequence):
        """
        Check if there are enough free blocks to allocate for the sequence.

        Args:
            sequence (Sequence): The sequence to allocate blocks for.
        """
        return len(self.free_block_ids) >= sequence.num_blocks

    def allocate(self, sequence):
        """
        Allocate blocks for the given sequence.

        Args:
            sequence (Sequence): The sequence to allocate blocks for.
        """
        assert not sequence.block_table
        h = -1
        cache_miss = False

        for i in range(sequence.num_blocks):
            token_ids = sequence.block(i)
            if len(token_ids) == self.block_size:
                # the i-th block is full, h is computed as a valid hash value
                h = self.compute_hash(token_ids, prefix=h)
            else:
                # otherwise, we set the h to -1 to indicate that this is an incomplete block.
                # we don't share incomplete partial blocks among sequences.
                h = -1

            block_id = self.hash_to_block_id.get(h, -1)
            is_incomplete_or_new_block = block_id == -1
            has_hash_collision = block_id != -1 and self.blocks[block_id].token_ids != token_ids

            if is_incomplete_or_new_block or has_hash_collision:
                # if cache_miss is set to True, we never change this back to False.
                # this means that once we have a cache miss for any block in the sequence,
                # all subsequent blocks will be treated as cache misses too.
                cache_miss = True

            if cache_miss:
                # we get a block id from free_block_ids and assign it to this block
                block_id = self.free_block_ids[0]
                block = self.allocate_block(block_id)
            else:
                # if this is full block:
                sequence.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # if the block is already used, we just increase its ref_count.
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # otherwise, we allocate it.
                    block = self.allocate_block(block_id)
            if h != -1:
                # if the hash value is valid, we update the hash_to_block_id mapping
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            sequence.block_table.append(block_id)

    def deallocate(self, sequence):
        """
        Deallocate all blocks used by the given sequence.

        Args:
            sequence (Sequence): The sequence to deallocate blocks for.
        """
        for block_id in reversed(sequence.block_table):
            block = self.blocks[block_id]
            assert block.ref_count > 0
            block.ref_count -= 1
            if block.ref_count == 0:
                # if no sequence is using this block, we can deallocate it
                self.deallocate_block(block_id)
        sequence.num_cached_tokens = 0
        sequence.block_table.clear()

    def can_append(self, sequence):
        """
        Check if there is enough space to append a token to the sequence.

        Args:
            sequence (Sequence): The sequence to check.

        Returns:
            bool: True if there is enough space to append, False otherwise.
        """
        num_tokens_in_last_block = len(sequence) % self.block_size
        need_new_block = 1 if num_tokens_in_last_block == 1 else 0
        # if num_tokens_in_last_block is 1, this means we just started a new block,
        # so we need to ensure that there is at least one free block available.
        return len(self.free_block_ids) >= need_new_block

    def may_append(self, sequence):
        """
        Update the block manager state after appending a token to the sequence.

        Args:
            sequence (Sequence): The sequence that has been appended to.
        """
        block_table = sequence.block_table
        last_block = self.blocks[block_table[-1]]
        num_tokens_in_last_block = len(sequence) % self.block_size

        if num_tokens_in_last_block == 1:
            # just starting a new block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self.allocate_block(block_id)
            block_table.append(block_id)
        elif num_tokens_in_last_block == 0:
            # last block is full, need to allocate a new block
            assert last_block.hash == -1
            token_ids = sequence.block(sequence.num_blocks - 1)
            # -2 means the previous block of the last block
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # last block is incomplete, do nothing
            assert last_block.hash == -1
