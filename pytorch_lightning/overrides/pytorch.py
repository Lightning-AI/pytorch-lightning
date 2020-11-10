import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp.grad_scaler import OptState
import torch.distributed as dist


class ShardedGradScaler(GradScaler):
    def step(self, optimizer, *args, **kwargs):
        if (not self._enabled):
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        retval = None

        if (hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling):
            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            retval = optimizer.step(*args, **dict(kwargs, grad_scaler=self))
            optimizer_state["stage"] = OptState.STEPPED
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

        num_infs = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())

        # Ensure we collect number of inf grads across all processes
        num_infs = torch.tensor(num_infs, dtype=torch.int, device=self._scale.device)
        if dist.is_initialized():
            dist.all_reduce(num_infs)
        if not num_infs.item() > 0:
            retval = optimizer.step(*args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval
