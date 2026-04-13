Distributed Timeline Parallel
=============================

The timeline parallel means each node in the graph performs independent computation
since each node has its own independent timeline. This parallel approach emphasizes that the timelines of each node are
independent and may span multiple machines or processes, and nodes may not immediately synchronize or communicate with each other.

Starygl provides SequencePipe class to help user implement timeline parallelism conveniently.The SequencePipe class provides methods and functions
that allow users to easily define and execute parallel operations. By inheriting the sequencepipe class,
users can easily implement their own models and accelerate the training process using timeline parallelism.

Here we provide an RNN example, user just need to let model inherit SequencePipe class without changing one line of code to
implement timeline parallel.

.. code-block:: python

    class SimpleRNN(SequencePipe):
    def __init__(self,
        num_classes: int,
        hidden_dims: int,
        num_layers: int,
        device: Any,
        group: Any,
    ) -> None:
        super().__init__()
        self.device = device
        self.group = group

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.gru = nn.GRU(
            input_size = hidden_dims,
            hidden_size = hidden_dims,
            num_layers = num_layers,
            batch_first = True,
        )
        self.out = nn.Linear(hidden_dims, num_classes)

    def forward(self, inputs, states):
        x, = inputs # (N, L, H)
        h, = states # (N, L, H)

        h = h.transpose(0, 1).contiguous() # (L, N, H)
        x, h = self.gru(x, h) # (N, L, H), (L, N, H)
        h = h.transpose(0, 1).contiguous() # (N, L, H)
        x = self.out(x)
        return (x,), (h, )

    def loss_fn(self, inputs, labels) -> Tensor:
        x, = inputs
        mask, y = labels

        x = x[mask, -1]
        if x.numel() > 0:
            y = y[mask]
            return F.cross_entropy(x, y)
        else:
            return x.mul(0.0).sum()

    def get_group(self) -> Any:
        return self.group

    def get_init_states(self):
        s = torch.zeros(self.num_layers, self.hidden_dims).to(self.device)
        return (s,)
