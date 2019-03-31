# Testing setup

## A. Enable CircleCI for your project
1. Integrate CircleCI by clicking "Set up Project" at [this link](https://circleci.com/add-projects/gh/NextGenVest).

## B. Add your own tests
1. In the /tests, emulate exactly the folder structure for your module found under /bot_seed
2. To create a test for file ```/bot_seed/folder/example.py```:
    - create the file ```/tests/folder/example_test.py```
    - notice the **_test**
    - notice the mirror path under **/tests**

3. Your ```example_test.py``` file should have these main components

```python
# example.py

def function_i_want_to_test(x):
    return x*2

def square(x):
    return x*x

```

```python
# example_test.py

import pytest

# do whatever imports you need
from app.bot_seed.folder.example import function_i_want_to_test, square

def test_function_i_want_to_test():
    answer = function_i_want_to_test(4)
    assert answer == 8

# -----------------------------------
# Your function must start with test_
# -----------------------------------
def test_square():
    answer = square(3)
    assert answer == 9

# -----------------------------------
# boilerplate (link this file to pytest)
# -----------------------------------
if __name__ == '__main__':
    pytest.main([__file__])
```

## C. Add build passing badge
1. Create a CircleCI status token:
    - Go here: https://circleci.com/gh/NextGenVest/your-project-name/edit#api
    - Click create token
    - Select status
    - Type "badge status"

2. Get a copy of the markdown code:
    - Go here: https://circleci.com/gh/NextGenVest/your-project-name/edit#badges
    - Select master
    - Select "badge status" token
    - Select image URL
    - Copy the image url link and change the html at the top of the root README.md file for your project
    