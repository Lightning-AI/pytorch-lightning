# Project requirements

This is the root requirements' folder which branches into sub-folders depending on python package.
Within the folder we have grouped requirements files/lists per focus which shall closely match package extra
So, for example when you install PL as `pip install pytorch-lightning[loggers]` this list is stored in `requirements/pytorch/loggers.txt`.
The only exceptional requirement file is `devel.txt` which is aggregated al needed requirements for development.

## CI/CD upper bounds

For Ci stability we have set for all package version upper bounds (the latest version) so with any sudden release we won't put our development in fire.
The continues updated of these apper bounds is managed by dependabot.
Note that ll of these upper bounds are lifter when you install package either from source or as a package.
In case, you want to preserve/enforce restriction on latest compatible version, add "strict" as in-line comment.
