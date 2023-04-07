import jax.numpy as jnp


class Sequential:
    def __init__(self, layers):
        """Initialize a sequential model of layers.

        Args:
            layers (list or tuple of Layers): List or tuple of layers for the feedforward model.
        """
        self.layers = layers

        self.states = [layer.init_state() for layer in layers]
        self.parameters = [layer.init_parameters() for layer in layers]

    def __call__(self, inputs: jnp.ndarray):
        for i in range(len(self.layers)):
            # TODO: values passed shouldn't be assumed to always be rates
            self.states[i].input_nodes.rate = inputs
            self.states[i] = self.layers[i].update_state(
                self.states[i], self.parameters[i]
            )
            inputs = self.states[i].output_nodes.rate

        return inputs
