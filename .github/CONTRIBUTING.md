# Contributing    
Welcome to the PyTorch Lightning community! We're building the most advanced research platform on the planet to implement the latest, best practices that the amazing PyTorch team rolls out!   

## Main Core Value: One less thing to remember

Simplify the API as much as possible from the user perspective.
 Any additions or improvements should minimize things the user needs to remember.   

For example: One benefit of the validation_step is that the user doesn't have to remember to set the model to .eval().
 This avoids all sorts of subtle errors the user could make.  

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
There are 1,000 ways to do something. However, something eventually becomes standard practice that everyone does.
 Thus we pick one way of doing it and force everyone to do it this way.
 A good example is accumulated gradients.
 There are many ways to implement, we just pick one and force users to use that one.
 A bad forced decision would be to make users use a specific library to do something.    

When something becomes a best practice, we add it to the framework. This likely looks like code in utils or in the model file that everyone keeps adding over and over again across projects. When this happens, bring that code inside the trainer and add a flag for it.

#### Simple External API
What makes sense to you may not make sense to others. Create an issue with an API change suggestion and validate that it makes sense for others.
 Treat code changes how you treat a startup: validate that it's a needed feature, then add if it makes sense for many people.

#### Backward-compatible API
We all hate updating our deep learning packages because we don't want to refactor a bunch of stuff. In Lightning, we make sure every change we make which could break an API is backwards compatible with good deprecation warnings.

**You shouldn't be afraid to upgrade Lightning :)**

#### Gain User Trust
As a researcher you can't have any part of your code going wrong. So, make thorough tests that ensure an implementation of a new trick or subbtle change is correct.

#### Interoperability
Have a favorite feature from other libraries like fast.ai or transformers? Those should just work with lightning as well. Grab your favorite model or learning rate scheduler from your favorite library and run it in Lightning.

---

## Contribution Types
Currently looking for help implementing new features or adding bug fixes.

A lot of good work has already been done in project mechanics (requirements.txt, setup.py, pep8, badges, ci, etc...) we're in a good state there thanks to all the early contributors (even pre-beta release)!

### Bug Fixes:
1. Submit a github issue - try to decried what happen so other can reproduce it too.
2. Try to ix it or recommend a solution...
3. Submit a PR!


### New Features:
1. Submit a github issue - describe what is motivation of such feature (plus an use-case).
2. Let's discuss to agree on the feature scope.
3. Submit a PR! (with updated docs and tests 🙃).

---

## Guidelines

### Coding Style

1. Use f-strings for output formation (except logging when we stay with lazy `logging.info("Hello %s!`, name).
2. Test the code with flake8, run locally PEP8 fixes:
    ```
    autopep8 -v -r --max-line-length 120 --in-place .
    ```

### Documentation

We are using Sphinx with Napoleon extension. 
Moreover we set Google style to follow with type convention. 

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
cd docs
pip install -r requirements.txt
make html
```
and open `docs/build/html/index.html` in your browser.

When you send a PR the continuous integration will run tests and build the docs. You can access a preview of the html pages in the 
_Artifacts_ tab in CircleCI when you click on the task named _ci/circleci: Build-Docs_ at the bottom of the PR page.

### Testing

Test your work locally to speed up your work since so you can focus only in particular (failing) test-cases.
 To setup a local development environment, install both local and test dependencies:
```bash
pip install -r requirements.txt
pip install -r tests/requirements-devel.txt
``` 

You can run the full test-case in your terminal via this bash script: 

```bash
bash .run_local_tests.sh
```

Note: if your computer does not have multi-GPU nor TPU these tests are skipped.

For convenience, you can use also your own CircleCI building which will be triggered with each commit.
This is useful if you do not test against all required dependencies version.
To do so, login to [CircleCI](https://app.circleci.com/) and enable your forked project in the dashboard. It will just work after that.

### Pull Request

We welcome any useful contribution! For convinece here's a recommended workflow:

0. Think about what you want to do - fix a bug, repair docs, etc. 
1. Start your work locally (usually until you need our CI testing)
    - create a branch and prepare your changes
    - hint: do not work with your master directly, it may become complicated when you need to rebase
    - hint: give your PR a good name! it will be useful later when you may work on multiple tasks/PRs
2. Create a "Draft PR" which is clearly marked which lets us know you don't need feedback yet.
3. When you feel like you are ready for integrating your work, turn your PR to "Ready for review".
4. Use tags in PR name for following cases:
    - **[blocked by #<number>]** if you work is depending on others changes
    - **[wip]** when you start to re-edit your work, mark it so no one will accidentally merge it in meantime

### Question & Answer

1. **How can I help/contribute?**

    All help is very welcome - reporting bug, solving issues and preparing bug fixes. To solve some issues you can start with label [good first issue](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or chose something close to your domain with label [help wanted](https://github.com/PyTorchLightning/pytorch-lightning/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22). Before you start to implement anything check that the issue description that it is clear and self-assign the task to you (if it is not possible, just comment that you take it and we assign it to you...).

2. **Is there a recommendation for branch names?**
    
    We do not rely on the name convention so far you are working with your own fork. Anyway it would be nice to follow this convention `<type>/<issue-id>_<short-name>` where the types are: `bugfix`, `feaure`, `docs`, `tests`, ...

3. **How to rebase my PR?**
    
    We recommend to create a PR in separate branch different from `master`, especially if you plan to submit several changes and do not want to wait until the fist one is resolved (we can work on them in parallel). Update your master with upstream (assuming you have already set [upstream](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork))
    ```bash
    git fetch --all --prune
    git checkout master
    git merge upstream/master
    ```
    checkout your feature branch
    ```bash
    git checkout my-PR-branch
    git rebase master
    # follow git instructions to resolve conflists
    git push -f
    ```
