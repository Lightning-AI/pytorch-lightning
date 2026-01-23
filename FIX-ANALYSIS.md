# PyTorch Lightning PR #21500 - Test Failure Analysis & Fix

## Root Cause Identified

The batch interval scheduler implementation has a design issue where the LR tracking in `on_train_batch_start` is called **BEFORE** the scheduler is stepped.

### The Timeline Issue

```
Loop iteration:
1. on_train_batch_start() ← LR is tracked here (BEFORE step)
2. training_step()
3. optimizer.step()
4. update_lr_schedulers("batch") ← Scheduler steps HERE
5. on_train_batch_end()
```

So the LR values recorded in `on_train_batch_start` are from the **previous** batch's scheduler step!

### Why Tests Fail

The test expects:
```python
assert model.lr_history[0] > model.lr_history[-1]  # LR should decrease
```

But because LR is tracked BEFORE the scheduler steps, all tracked LRs are shifted by one batch, causing the test assertion to fail.

## Solution

Move the batch scheduler update to BEFORE `on_train_batch_start` is called, or track LR at a different point.

## Best Fix Approach

Update the batch scheduler BEFORE on_train_batch_start:

1. Move `update_lr_schedulers("batch", ...)` to occur BEFORE the training_step
2. This ensures LR changes are reflected in on_train_batch_start hook
3. Maintains consistency with "step" interval behavior

## Implementation

The batch interval scheduler should be called at the beginning of the batch loop, not after the optimizer step.

Current flow (WRONG):
```
batch_start_hook()
  → training_step()
    → optimizer.step()
  → update_schedulers("batch")  ← Too late!
batch_end_hook()
```

Should be (CORRECT):
```
update_schedulers("batch")  ← Update first
batch_start_hook()
  → training_step()
    → optimizer.step()
batch_end_hook()
```

## Files to Modify

File: `src/lightning/pytorch/loops/training_epoch_loop.py`

Changes needed:
1. Move batch scheduler update to earlier in the loop
2. Ensure it runs before on_train_batch_start hook
3. Keep epoch/step interval logic unchanged
