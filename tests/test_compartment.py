import jax.numpy as jnp
import plastix as px
import unittest


class TestCompartment(unittest.TestCase):
    def test_and(self):
        n = 2
        m = 1
        # dense layer as sparse layer
        edges = [(0, 0), (1, 0)]

        layer = px.layers.SparseLayer(
            n,
            m,
            edges,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.CompartmentKernel(),
        )

        state = layer.init_state()
        parameters = layer.init_parameters()
        parameters.edges.weight *= 0.5

        # 0, 0 -> 0

        state.input_nodes.rate = jnp.array([[0.0], [0.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] == 0

        # 0, 1 -> <0.5

        state.input_nodes.rate = jnp.array([[0.0], [1.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] < 0.5

        # 1, 0 -> <0.5

        state.input_nodes.rate = jnp.array([[1.0], [0.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] < 0.5

        # 1, 1 -> >0.5

        state.input_nodes.rate = jnp.array([[1.0], [1.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] > 0.5

        state.output_nodes.dendritic_input = jnp.array([0, -1, 0, 1, 0, 0])
        state = layer.update_state(state, parameters)
