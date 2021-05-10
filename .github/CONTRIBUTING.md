# Contributing

Welcome to the PyTorch Lightning community! We're building the most advanced research platform on the planet to implement the latest, best practices that the amazing PyTorch team rolls out!

## Main Core Value: One less thing to remember

Simplify the API as much as possible from the user perspective.
Any additions or improvements should minimize the things the user needs to remember.

For example: One benefit of the validation_step is that the user doesn't have to remember to set the model to .eval().
This helps users avoid all sorts of subtle errors.

## Lightning Design Principles

We encourage all sorts of contributions you're interested in adding! When coding for lightning, please follow these principles.

#### No PyTorch Interference

We don't want to add any abstractions on top of pure PyTorch.
This gives researchers all the control they need without having to learn yet another framework.

#### Simple Internal Code

It's useful for users to look at the code and understand very quickly what's happening.
Many users won't be engineers. Thus we need to value clear, simple code over condensed ninja moves.
While that's super cool, this isn't the project for that :)

#### Force User Decisions To Best Practices

There are 1,000 ways to do something. However, eventually one popular solution becomes standard practice, and everyone follows.
We try to find the best way to solve a particular problem, and then force our users to use it for readability and simplicity.
A good example is accumulated gradients.
There are many different ways to implement it, we just pick one and force users to use it.
A bad forced decision would be to make users use a specific library to do something.

When something becomes a best practice, we add it to the framework. This is usually something like bits of code in utils or in the model file that everyone keeps adding over and over again across projects. When this happens, bring that code inside the trainer and add a flag for it.

#### Simple External API

What makes sense to you may not make sense to others. When creating an issue with an API change suggestion, please validate that it makes sense for others.
Treat code changes the way you treat a startup: validate that it's a needed feature, then add if it makes sense for many people.

#### Backward-compatible API

We all hate updating our deep learning packages because we don't want to refactor a bunch of stuff. In Lightning, we make sure every change we make which could break an API is backward compatible with good deprecation warnings.

**You shouldn't be afraid to upgrade Lightning :)**

#### Gain User Trust

As a researcher, you can't have any part of your code going wrong. So, make thorough tests to ensure that every implementation of a new trick or subtle change is correct.

#### Interoperability

Have a favorite feature from other libraries like fast.ai or transformers? Those should just work with lightning as well. Grab your favorite model or learning rate scheduler from your favorite library and run it in Lightning.

---

## Contribution Types

We are always looking for help implementing new features or fixing bugs.

A lot of good work has already been done in project mechanics (requirements.txt, setup.py, pep8, badges, ci, etc...) so we're in a good state there thanks to all the early contributors (even pre-beta release)!

### Bug Fixes:

1. If you find a bug please submit a github issue.

   - Make sure the title explains the issue.
   - Describe your setup, what you are trying to do, expected vs. actual behaviour. Please add configs and code samples.
   - Add details on how to reproduce the issue - a minimal test case is always best, colab is also great.
     Note, that the sample code shall be minimal and if needed with publicly available data.

2. Try to fix it or recommend a solution. We highly recommend to use test-driven approach:

   - Convert your minimal code example to a unit/integration test with assert on expected results.
   - Start by debugging the issue... You can run just this particular test in your IDE and draft a fix.
   - Verify that your test case fails on the master branch and only passes with the fix applied.

3. Submit a PR!

_**Note**, even if you do not find the solution, sending a PR with a test covering the issue is a valid contribution and we can help you or finish it with you :]_

### New Features:

1. Submit a github issue - describe what is the motivation of such feature (adding the use case or an example is helpful).
2. Let's discuss to determine the feature scope.
3. Submit a PR! We recommend test driven approach to adding new features as well:

   - Write a test for the functionality you want to add.
   - Write the functional code until the test passes.

4. Add/update the relevant tests!

- [This PR](https://github.com/PyTorchLightning/pytorch-lightning/pull/2671) is a good example for adding a new metric, and [this one for a new logger](https://github.com/PyTorchLightning/pytorch-lightning/pull/2721).

### Test cases:

Want to keep Lightning healthy? Love seeing those green tests? So do we! How to we keep it that way? We write tests! We value tests contribution even more than new features.

Most of the tests in PyTorch Lightning train a trial MNIST model under various trainer conditions (ddp, ddp2+amp, etc...). The tests expect the model to perform to a reasonable degree of testing accuracy to pass. Want to add a new test case and not sure how? [Talk to us!](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ)

---

## Guidelines

### Developments scripts
To build the documentation locally, simply execute the following commands from project root (only for Unix):
- `make clean` cleans repo from temp/generated files
- `make docs` builds documentation under _docs/build/html_
- `make test` runs all project's tests with coverage

### Original code

All added or edited code shall be the own original work of the particular contributor.
If you use some third-party implementation, all such blocks/functions/modules shall be properly referred and if possible also agreed by code's author. For example - `This code is inspired from http://...`.
In case you adding new dependencies, make sure that they are compatible with the actual PyTorch Lightning license (ie. dependencies should be _at least_ as permissive as the PyTorch Lightning license).

### Coding Style

1. Use f-strings for output formation (except logging when we stay with lazy `logging.info("Hello %s!", name)`.
2. You can use `pre-commit` to make sure your code style is correct.

### Documentation

We are using Sphinx with Napoleon extension.
Moreover, we set Google style to follow with type convention.

- [Napoleon formatting with Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [ReStructured Text (reST)](https://docs.pylonsproject.org/projects/docs-style-guide/)
- [Paragraph-level markup](https://www.sphinx-doc.org/en/1.5/markup/para.html)

See following short example of a sample function taking one position string and optional

```python
from typing import Optional

def my_func(param_a: int, param_b: Optional[float] = None) -> str:
    """Sample function.

    Args:
        param_a: first parameter
        param_b: second parameter

    Return:
        sum of both numbers

    Example:
        Sample doctest example...
        >>> my_func(1, 2)
        3

    .. note:: If you want to add something.
    """
    p = param_b if param_b else 0
    return str(param_a + p)
```

When updating the docs make sure to build them first locally and visually inspect the html files (in the browser) for
formatting errors. In certain cases, a missing blank line or a wrong indent can lead to a broken layout.
Run these commands

```bash
pip install -r requirements/docs.txt
cd docs
make html
```

and open `docs/build/html/index.html` in your browser.

Notes:

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive by doing one of the following:
  - on Ubuntu (Linux) run `apt-get install texlive` or otherwise follow the instructions on the TeXLive website
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)
- with PL used class meta you need to use python 3.7 or higher

When you send a PR the continuous integration will run tests and build the docs. You can access a preview of the html pages in the
_Artifacts_ tab in CircleCI when you click on the task named _ci/circleci: Build-Docs_ at the bottom of the PR page.

### Testing

**Local:** Testing your work locally will help you speed up the process since it allows you to focus on particular (failing) test-cases.
To setup a local development environment, install both local and test dependencies:

```bash
python -m pip install ".[dev, examples]"
python -m pip install pre-commit
```

You can run the full test-case in your terminal via this make script:

```bash
make test
```

Note: if your computer does not have multi-GPU nor TPU these tests are skipped.

**GitHub Actions:** For convenience, you can also use your own GHActions building which will be triggered with each commit.
This is useful if you do not test against all required dependency versions.

**Docker:** Another option is utilize the [pytorch lightning cuda base docker image](https://hub.docker.com/repository/docker/pytorchlightning/pytorch_lightning/tags?page=1&name=cuda). You can then run:

```bash
python -m pytest pytorch_lightning tests pl_examples -v
```

You can also run a single test as follows:

```bash
python -m pytest -v tests/trainer/test_trainer_cli.py::test_default_args
```

### Pull Request

We welcome any useful contribution! For your convenience here's a recommended workflow:

0. Think about what you want to do - fix a bug, repair docs, etc. If you want to implement a new feature or enhance an existing one, start by opening a GitHub issue to explain the feature and the motivation. Members from core-contributors will take a look (it might take some time - we are often overloaded with issues!) and discuss it. Once an agreement was reached - start coding.
1. Start your work locally (usually until you need our CI testing).
   - Create a branch and prepare your changes.
   - Tip: do not work with your master directly, it may become complicated when you need to rebase.
   - Tip: give your PR a good name! It will be useful later when you may work on multiple tasks/PRs.
2. Test your code!
   - It is always good practice to start coding by creating a test case, verifying it breaks with current behaviour, and passes with your new changes.
   - Make sure your new tests cover all different edge cases.
   - Make sure all exceptions are handled.
3. Create a "Draft PR" which is clearly marked, to let us know you don't need feedback yet.
4. When you feel ready for integrating your work, mark your PR "Ready for review".
   - Your code should be readable and follow the project's design principles.
   - Make sure all tests are passing.
   - Make sure you add a GitHub issue to your PR.
5. Use tags in PR name for following cases:
   - **[blocked by #<number>]** if you work is depending on others changes.
   - **[wip]** when you start to re-edit your work, mark it so no one will accidentally merge it in meantime.

### Question & Answer

#### How can I help/contribute?

All types of contributions are welcome - reporting bugs, fixing documentation, adding test cases, solving issues, and preparing bug fixes.
To get started with code contributions, look for issues marked with the label [good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or chose something close to your domain with the label [help wanted](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22). Before coding, make sure that the issue description is clear and comment on the issue so that we can assign it to you (or simply self-assign if you can).

#### Is there a recommendation for branch names?

We recommend you follow this convention `<type>/<issue-id>_<short-name>` where the types are: `bugfix`, `feature`, `docs`, or `tests` (but if you are using your own fork that's optional).

#### How to rebase my PR?

We recommend creating a PR in a separate branch other than `master`, especially if you plan to submit several changes and do not want to wait until the first one is resolved (we can work on them in parallel).

First, make sure you have set [upstream](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork) by running:

```bash
git remote add upstream https://github.com/PyTorchLightning/pytorch-lightning.git
```

You'll know its set up right if you run `git remote -v` and see something similar to this:

```bash
origin  https://github.com/{YOUR_USERNAME}/pytorch-lightning.git (fetch)
origin  https://github.com/{YOUR_USERNAME}/pytorch-lightning.git (push)
upstream        https://github.com/PyTorchLightning/pytorch-lightning.git (fetch)
upstream        https://github.com/PyTorchLightning/pytorch-lightning.git (push)
```

Checkout your feature branch and rebase it with upstream's master before pushing up your feature branch:

```bash
git fetch --all --prune
git rebase upstream/master
# follow git instructions to resolve conflicts
git push -f
```

#### How to add new tests?**

We are using [pytest](https://docs.pytest.org/en/stable/) in Pytorch Lightning.

Here are tutorials:
* (recommended) [Visual Testing with pytest](https://www.youtube.com/playlist?list=PLCTHcU1KoD99Rim2tzg-IhYY2iu9FFvNo) from JetBrains on YouTube
* [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/) article on realpython.com

Here is the process to create a new test

* 0. Optional: Follow tutorials !
* 1. Find a file in tests/ which match what you want to test. If none, create one.
* 2. Use this template to get started !
* 3. Use `BoringModel and derivates to test out your code`.

```python
# TEST SHOULD BE IN YOUR FILE: tests/..../...py
# TEST CODE TEMPLATE

# [OPTIONAL] pytest decorator
# @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_explain_what_is_being_tested(tmpdir):
    """
    Test description about text reason to be
    """

    # os.environ["PL_DEV_DEBUG"] = '1' # [OPTIONAL] When activated, you can use internal trainer.dev_debugger

    class ExtendedModel(BoringModel):
        ...

    model = ExtendedModel()

    # BoringModel is a functional model. You might want to set methods to None to test your behaviour
    # Example: model.training_step_end = None

    trainer = Trainer(
        default_root_dir=tmpdir, # will save everything within a tmpdir generated for this test
        ...
    )
    trainer.fit(model)
    trainer.test()  # [OPTIONAL]

    # assert the behaviour is correct.
    assert ...
```
run our/your test with
```bash
python -m pytest tests/..../...py::test_explain_what_is_being_tested --verbose --capture=no
```


#### How to fix PR with mixed base and target branches?

Sometimes you start your PR as a bug-fix but it turns out to be more of a feature (or the other way around).
Do not panic, the solution is very straightforward and quite simple.
All you need to do are these two steps in arbitrary order:
   - Ask someone from Core to change the base/target branch to the correct one
   - Rebase or cherry-pick your commits onto the correct base branch...

Let's show how to deal with the git...
the sample case is moving a PR from `master` to `release/1.2-dev` assuming my branch name is `my-branch`
and the last true master commit is `ccc111` and your first commit is `mmm222`.
   * **Cherry-picking** way
     ```bash
     git checkout my-branch
     # create a local backup of your branch
     git checkout -b my-branch-backup
     # reset your branch to the correct base
     git reset release/1.2-dev --hard
     # ACTION: this step is much easier to do with IDE
     #  so open one and cherry-pick your last commits from `my-branch-backup`
     #  resolve all eventual conflict as the new base may contain different code
     # when all done, push back to the open PR
     git push -f
     ```
   * **Rebasing way**, see more about [rebase onto usage](https://womanonrails.com/git-rebase-onto)
     ```bash
     git checkout my-branch
     # rebase your commits on the correct branch
     git rebase --onto release/1.2-dev ccc111
     # if there is no collision you shall see just success
     #  eventually you would need to resolve collision and in such case follow the instruction in terminal
     # when all done, push back to the open PR
     git push -f
     ```


### Bonus Workflow Tip

If you don't want to remember all the commands above every time you want to push some code/setup a Lightning Dev environment on a new VM, you can set up bash aliases for some common commands. You can add these to one of your `~/.bashrc`, `~/.zshrc`, or `~/.bash_aliases` files.

NOTE: Once you edit one of these files, remember to `source` it or restart your shell. (ex. `source ~/.bashrc` if you added these to your `~/.bashrc` file).

```bash
plclone (){
    git clone https://github.com/{YOUR_USERNAME}/pytorch-lightning.git
    cd pytorch-lightning
    git remote add upstream https://github.com/PyTorchLightning/pytorch-lightning.git
    # This is just here to print out info about your remote upstream/origin
    git remote -v
}

plfetch (){
    git fetch --all --prune
    git checkout master
    git merge upstream/master
}

# Rebase your branch with upstream's master
# plrebase <your-branch-name>
plrebase (){
    git checkout $@
    git rebase master
}
```

Now, you can:

- clone your fork and set up upstream by running `plclone` from your terminal
- fetch upstream and update your local master branch with it by running `plfetch`
- rebase your feature branch (after running `plfetch`) by running `plrebase your-branch-name`
