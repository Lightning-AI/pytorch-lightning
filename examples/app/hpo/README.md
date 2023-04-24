# Build a Lightning Hyperparameter Optimization (HPO) App

## A bit of background

Traditionally, developing machine learning (ML) products requires choosing among a large space of
hyperparameters while creating and training the ML models. Hyperparameter optimization
(HPO) aims to find a well-performing hyperparameter configuration for a given ML model
on a dataset at hand, including the ML model,
its hyperparameters, and other data processing steps.

HPOs free the human expert from a tedious and error-prone, manual hyperparameter tuning process.

As an example, in the famous [scikit-learn](https://scikit-learn.org/stable/) library,
hyperparameters are passed as arguments to the constructor of
the estimator classes such as `C` kernel for
[Support Vector Classifier](https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm), etc.

It is possible and recommended to search the hyperparameter space for the best validation score.

An HPO search consists of:

- an objective method
- a defined parameter space
- a method for searching or sampling candidates

A naive method for sampling candidates is grid search, which exhaustively considers all
hyperparameter combinations from a user-specified grid.

Fortunately, HPO is an active area of research, and many methods have been developed to
optimize the time required to get strong candidates.

In the following tutorial, you will learn how to use Lightning together with [Optuna](https://optuna.org/).

[Optuna](https://optuna.org/) is an open source HPO framework to automate hyperparameter search.
Out-of-the-box, it provides efficient algorithms to search large spaces and prune unpromising trials for faster results.

First, you will learn about the best practices on how to implement HPO without the Lightning Framework.
Secondly, we will dive into a working HPO application with Lightning, and finally create a neat
[HiPlot UI](https://facebookresearch.github.io/hiplot/_static/demo/demo_basic_usage.html?hip.filters=%5B%5D&hip.color_by=%22dropout%22&hip.PARALLEL_PLOT.order=%5B%22uid%22%2C%22dropout%22%2C%22lr%22%2C%22loss%22%2C%22optimizer%22%5D)
for our application.

## Getting started

### Step 1: Download the data

```bash
python download_data.py
```

### Step 2: Run the HPO Lightning App without an UI

```bash
lightning run app app_wo_ui.py
```

### Step 3: Run the HPO Lightning App with HiPlot UI in Streamlit.

```bash
lightning run app app_wi_ui.py
```

## Learn More

In the documentation, search for `Build a Sweep App`.
