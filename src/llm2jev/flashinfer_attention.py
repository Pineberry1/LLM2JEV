"""FlashInfer's two-level cascade API for shared-prefix branch decoding.

The shared prefix occupies the first set of pages and every branch has its own
suffix pages. Plans and packed prefix pages are reusable across model layers
and repeated decisions on the same state. FlashInfer is an optional dependency.
"""

import math

import torch


class FlashInferSharedPrefixAttention:
    def __init__(
        self, batch: int, q_heads: int, kv_heads: int, head_dim: int,
        prefix_len: int, max_suffix_len: int, dtype: torch.dtype,
        device: torch.device, page_size: int = 16,
    ):
        try:
            import flashinfer
        except ImportError as exc:
            raise ImportError("Install flashinfer-python to use the FlashInfer backend") from exc
        if batch < 1 or prefix_len < 1 or max_suffix_len < 1 or q_heads % kv_heads:
            raise ValueError("Invalid batch, sequence lengths, or GQA head counts")
        if dtype not in (torch.float16, torch.bfloat16) or device.type != "cuda":
            raise ValueError("FlashInfer requires CUDA float16 or bfloat16 tensors")
        self.flashinfer = flashinfer
        self.batch = batch
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.prefix_pages = math.ceil(prefix_len / page_size)
        self.pages_per_branch = math.ceil(max_suffix_len / page_size)
        self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._wrappers = {}
        self._layer_caches = None
        self._cache_generation = None

    def new_cache(self, prefix_keys: torch.Tensor, prefix_values: torch.Tensor) -> torch.Tensor:
        expected = (self.prefix_len, self.kv_heads, self.head_dim)
        if prefix_keys.shape != expected or prefix_values.shape != expected:
            raise ValueError("Prefix K/V shape must match the runner")
        cache = torch.empty(
            (self.prefix_pages + self.batch * self.pages_per_branch, 2,
             self.page_size, self.kv_heads, self.head_dim),
            dtype=self.dtype, device=self.device,
        )
        self._write_prefix(cache, prefix_keys, prefix_values)
        return cache

    def _write_prefix(self, cache: torch.Tensor, prefix_keys: torch.Tensor,
                      prefix_values: torch.Tensor) -> None:
        padded_keys = torch.empty(
            (self.prefix_pages * self.page_size, self.kv_heads, self.head_dim),
            dtype=self.dtype, device=self.device,
        )
        padded_values = torch.empty_like(padded_keys)
        padded_keys[:self.prefix_len].copy_(prefix_keys)
        padded_values[:self.prefix_len].copy_(prefix_values)
        cache[:self.prefix_pages, 0].copy_(
            padded_keys.view(self.prefix_pages, self.page_size, self.kv_heads, self.head_dim)
        )
        cache[:self.prefix_pages, 1].copy_(
            padded_values.view(self.prefix_pages, self.page_size, self.kv_heads, self.head_dim)
        )

    def prepare_layer_caches(self, prefix_keys: list[torch.Tensor],
                             prefix_values: list[torch.Tensor],
                             generation: int) -> list[torch.Tensor]:
        if len(prefix_keys) != len(prefix_values):
            raise ValueError("Prefix layer counts differ")
        if self._layer_caches is None:
            self._layer_caches = [
                self.new_cache(k, v) for k, v in zip(prefix_keys, prefix_values)
            ]
        elif generation != self._cache_generation:
            for cache, k, v in zip(self._layer_caches, prefix_keys, prefix_values):
                self._write_prefix(cache, k, v)
        self._cache_generation = generation
        return self._layer_caches

    def append(self, cache: torch.Tensor, position: int, keys: torch.Tensor,
               values: torch.Tensor) -> None:
        if position < 0 or position >= self.pages_per_branch * self.page_size:
            raise ValueError("Suffix position exceeds allocated pages")
        if keys.shape != (self.batch, self.kv_heads, self.head_dim) or values.shape != keys.shape:
            raise ValueError("K/V must have shape [branches, kv_heads, head_dim]")
        page, offset = divmod(position, self.page_size)
        branch_pages = cache[self.prefix_pages + page::self.pages_per_branch]
        branch_pages[:, 0, offset].copy_(keys)
        branch_pages[:, 1, offset].copy_(values)

    def _plan(self, lengths: tuple[int, ...]):
        if lengths not in self._wrappers:
            if len(lengths) != self.batch or any(
                length < 1 or length > self.pages_per_branch * self.page_size
                for length in lengths
            ):
                raise ValueError("Each suffix length must be within the cache capacity")
            indptr = [0]
            indices = []
            for branch, length in enumerate(lengths):
                count = math.ceil(length / self.page_size)
                indices.extend(
                    self.prefix_pages + branch * self.pages_per_branch + page
                    for page in range(count)
                )
                indptr.append(len(indices))
            def i32(values):
                return torch.tensor(values, dtype=torch.int32, device=self.device)
            wrapper = self.flashinfer.MultiLevelCascadeAttentionWrapper(
                2, self.workspace, "NHD",
            )
            wrapper.plan(
                [i32([0, self.batch]), i32(range(self.batch + 1))],
                [i32([0, self.prefix_pages]), i32(indptr)],
                [i32(range(self.prefix_pages)), i32(indices)],
                [i32([(self.prefix_len - 1) % self.page_size + 1]),
                 i32([(length - 1) % self.page_size + 1 for length in lengths])],
                self.q_heads, self.kv_heads, self.head_dim, self.page_size,
                q_data_type=self.dtype, kv_data_type=self.dtype,
            )
            self._wrappers[lengths] = wrapper
        return self._wrappers[lengths]

    def forward(self, queries: torch.Tensor, cache: torch.Tensor,
                lengths: int | tuple[int, ...]) -> torch.Tensor:
        if isinstance(lengths, int):
            lengths = (lengths,) * self.batch
        if queries.shape != (self.batch, self.q_heads, self.head_dim):
            raise ValueError("Q shape must match the runner")
        return self._plan(lengths).run(queries, cache)
