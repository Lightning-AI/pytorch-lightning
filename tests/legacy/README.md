# Maintaining back-compatibility with come legacy versions

The aim of this section is set some baselines and workflows/guidelines for maintaining back compatibility with some legacies version of PL

At this moment we focus on ability running old checkpoints, so the flow here is to create a checkpoint with every release and store it in our public AWS storage and so each CI testing will pull this archive and test loading and resuming training with this model.

If you want to pull all saved version-checkpoints for local testing/development, call

```bash
bash .actions/pull_legacy_checkpoints.sh
```

To back populate collection with past version you can use following bash:

```bash
bash generate_checkpoints.sh "1.3.7" "1.3.8"
zip -r checkpoints.zip checkpoints/
```
