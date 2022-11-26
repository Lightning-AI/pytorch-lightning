# Maintaining backward compatibility with legacy versions

The aim of this section is to set some baselines and workflows/guidelines for maintaining backward compatibility with some legacy versions of PyTorch Lightning.

At this moment, we focus on ability to run old checkpoints, so the flow here is to create a checkpoint with every release and store it in our public AWS storage. Stored legacy checkpoints are then used in each CI to test loading and resuming training with the archived checkpoints.

## Download legacy checkpoints

If you want to pull all saved version-checkpoints for local testing/development, call

```bash
bash .actions/pull_legacy_checkpoints.sh
```

## Generate legacy checkpoints locally

To back populate collection with past versions you can use the following command:

```bash
bash generate_checkpoints.sh "1.3.7" "1.3.8"
zip -r checkpoints.zip checkpoints/
```
