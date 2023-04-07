class Tree:
    def __init__(self, input_layers, upstream_layers):
        """Initialize a tree model of layers, where branches and leaves are coalesced at each layer call.

        Args:
            input_layers (list or tuple of Layers): List or tuple of layers for input to the tree model (e.g. synapses).
            upstream_layers (list or tuple of Layers): List or tuple of layers for propagating signals down the tree (e.g. upstream dendritic branches). Should be 1 less layer than input_layers.
        """
        assert (
            len(input_layers) == len(upstream_layers) + 1
        ), "Must be one less upstream layer than input layer."
        self.input_layers = input_layers
        self.upstream_layers = upstream_layers

        self.input_states = [layer.init_state() for layer in input_layers]
        self.input_parameters = [layer.init_parameters() for layer in input_layers]

        self.upstream_states = [layer.init_state() for layer in upstream_layers]
        self.upstream_parameters = [
            layer.init_parameters() for layer in upstream_layers
        ]

    def __call__(self, inputs: list or tuple):
        self.input_states[0].input_nodes.rate = inputs[0]
        self.input_states[0] = self.input_layers[0].update_state(
            self.input_states[0], self.input_parameters[0]
        )
        upstream = self.input_states[0].output_nodes.rate
        for i in range(len(self.input_layers) - 1):
            # TODO: values passed shouldn't be assumed to always be rates
            # First input layer
            self.input_states[i + 1].input_nodes.rate = inputs[i + 1]
            self.input_states[i + 1] = self.input_layers[i + 1].update_state(
                self.input_states[i + 1], self.input_parameters[i + 1]
            )

            # Add new signal to upstream aignals
            upstream += self.input_states[i + 1].output_nodes.rate

            # Simulate passage of signal to next branch
            self.upstream_layers[i].input_nodes.rate = upstream
            self.upstream_states[i] = self.upstream_layers[i].update_state(
                self.upstream_states[i], self.upstream_parameters[i]
            )
            upstream = self.upstream_states[i].output_nodes.rate

        return upstream
