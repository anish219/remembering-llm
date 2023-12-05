import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    """
    Params
     - start_size: The size of the attention sink
     - recent_size: The size of the sliding window
     - context_size: The size of the context
     - cache_size: The total size of the cache
     - k_seq_dim: The dimension of the key in the tensor
     - v_seq_dim: The dimension of the value in the tensor
     - k_slice: The function to slice the keys
     - v_slice: The function to slice the values
     - last_saved: The last saved index within the sequence length dimension
     - context_added: Whether or not the context section of the KV Cache has been added
    """
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        context_size=300,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.context_size = context_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.last_saved = 0
        self.context_added = False

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        # no need to evict
        if seq_len <= self.cache_size:
            return past_key_values
        # evict seq_len - self.cache_size
        self.last_saved -= seq_len - self.cache_size
        # add the context section if not added yet
        if not self.context_added:
            self.context_added = True
            self.cache_size += self.context_size
            context_shape = past_key_values[-1][-1].size()
            context_shape = context_shape[:self.k_seq_dim] + (self.context_size) + context_shape[self.k_seq_dim+1:]
            return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            torch.zeros(context_shape),
                            self.k_slice(k, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.k_seq_dim
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            torch.zeros(context_shape),
                            self.v_slice(v, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size + self.context_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size + self.context_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


    def add_context(self, past_key_values, context):
        """
        Add the context tokens to the past_key_values and update
        the size of the context tokens within the kv cache
        """
        if past_key_values is None:
            return None
        assert(context[-1][-1].size(self.k_seq_dim) == self.context_size)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        k2,
                        self.k_slice(k, self.start_size + self.context_size, seq_len)
                    ],
                    dim=self.k_seq_dim,
                ),
                context,
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        v2,
                        self.k_slice(v, self.start_size + self.context_size, seq_len)
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for (k, v), (k2, v2) in zip(past_key_values, context)
        ]


    def get_new_kvs(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        ret = [
            [
                self.k_slice(k, self.last_saved, seq_len),
                self.v_slice(v, self.last_saved, seq_len),
            ]
            for k, v in past_key_values
        ]
        self.last_saved = seq_len
        return ret


    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        # evict seq_len + num_coming - self.cache_size
        self.last_saved -= seq_len + num_coming - self.cache_size
        if not self.context_added:
            self.context_added = True
            self.cache_size += self.context_size
            context_shape = past_key_values[-1][-1].size()
            context_shape = context_shape[:self.k_seq_dim] + (self.context_size) + context_shape[self.k_seq_dim+1:]
            return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            torch.zeros(context_shape),
                            self.k_slice(k, seq_len - self.recent_size + num_coming, seq_len),
                        ],
                        dim=self.k_seq_dim
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            torch.zeros(context_shape),
                            self.v_slice(v, seq_len - self.recent_size + num_coming, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size + self.context_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size + self.context_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        self.last_saved -= end - start
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def combine_kvs(self, kv1, kv2):
        """
        Combine kv1 and kv2
        If kv1 is None, just return kv2
        """
        if kv1 is None:
            return kv2
        return [
            [
                torch.cat(
                    [
                        k1, k2
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        v1, v2
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for (k1, v1), (k2, v2) in zip(kv1, kv2)
        ]