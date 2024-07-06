import os
import random
import unittest

import torch

# Only import numpy if it's available
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from lightning.fabric.utilities.seed import (
    _collect_rng_states,
    _set_rng_states,
    pl_worker_init_function,
    reset_seed,
    seed_everything,
)


class TestSeedingFunctions(unittest.TestCase):
    def test_seed_everything(self):
        seed = 42
        seed_everything(seed)

        self.assertEqual(int(os.environ["PL_GLOBAL_SEED"]), seed)

        # Verify random seed by checking the state directly
        random_state = random.getstate()

        # Verify torch seed by checking tensor value
        torch.manual_seed(seed)
        expected_tensor = torch.randn(1)
        torch.manual_seed(seed)
        test_tensor = torch.randn(1)
        self.assertTrue(torch.equal(expected_tensor, test_tensor))

        if _NUMPY_AVAILABLE:
            np.random.seed(seed)
            expected_np_value = np.random.rand()
            np.random.seed(seed)
            test_np_value = np.random.rand()
            self.assertEqual(expected_np_value, test_np_value)

        seed_everything(seed)
        self.assertEqual(random.getstate(), random_state)

    def test_reset_seed(self):
        original_seed = 123
        seed_everything(original_seed)

        random_state = random.getstate()
        torch_state = torch.get_rng_state()

        if _NUMPY_AVAILABLE:
            np_state = np.random.get_state()

        reset_seed()

        self.assertEqual(int(os.environ["PL_GLOBAL_SEED"]), original_seed)
        self.assertEqual(random.getstate(), random_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

        if _NUMPY_AVAILABLE:
            self.assertTrue(np.array_equal(np.random.get_state(), np_state))

    def test_pl_worker_init_function(self):
        seed = 42
        seed_everything(seed)
        worker_id = 0

        # Capture the RNG states before initializing the worker
        before_states = _collect_rng_states()

        pl_worker_init_function(worker_id)

        # Capture the RNG states after initializing the worker
        after_states = _collect_rng_states()

        self.assertFalse(torch.equal(before_states["torch"], after_states["torch"]))
        self.assertNotEqual(before_states["python"], after_states["python"])

        if _NUMPY_AVAILABLE:
            self.assertFalse(np.array_equal(before_states["numpy"][1], after_states["numpy"][1]))

    def test_collect_and_set_rng_states(self):
        seed = 42
        seed_everything(seed)

        states = _collect_rng_states()
        self.assertIn("torch", states)
        self.assertIn("python", states)

        if _NUMPY_AVAILABLE:
            self.assertIn("numpy", states)
        if torch.cuda.is_available():
            self.assertIn("torch.cuda", states)

        new_seed = 123
        seed_everything(new_seed)

        _set_rng_states(states)

        restored_states = _collect_rng_states()
        self.assertTrue(torch.equal(torch.tensor(states["torch"]), torch.tensor(restored_states["torch"])))
        self.assertEqual(states["python"], restored_states["python"])

        if _NUMPY_AVAILABLE:
            np.testing.assert_array_equal(states["numpy"][1], restored_states["numpy"][1])
        if torch.cuda.is_available():
            for state1, state2 in zip(states["torch.cuda"], restored_states["torch.cuda"]):
                self.assertTrue(torch.equal(torch.tensor(state1), torch.tensor(state2)))


if __name__ == "__main__":
    unittest.main()
