from plastix.kernels import State, Parameter
from plastix.kernels.nodes import NodeKernel
import jax.numpy as jnp


class CompartmentKernel(NodeKernel):
    potential = State((1,), jnp.zeros)
    dendritic_input = State((1,), jnp.zeros)
    bias = Parameter((1,), jnp.zeros)

    def update_state(self, edges):
        activation = (
            (jnp.sign(self.dendritic_input) / 2 + 0.5)
            * jnp.sum(edges.weighted_potential)
            + self.dendritic_input
            + self.bias
        )
        self.potential = jnp.tanh(activation)

    def update_parameters(self, edges):
        pass
