# Project Requirements

This root requirements folder branches into sub-folders depending on the python package.
Within the folder, we have grouped requirements files/lists per focus, which shall closely match package extra
So, for example, when you install PL as `pip install pytorch-lightning[extra]`, this list is stored in `requirements/pytorch/extra.txt`.

## CI/CD upper bounds

For Ci stability, we have set for all package versions upper bounds (the latest version), so with any sudden release, we won't put our development on fire.
The continues updated of these upper bounds are managed by dependabot.
Note that these upper bounds are lifters when installing a package from the source or as a package.
If you want to preserve/enforce restrictions on the latest compatible version, add "strict" as an in-line comment.
