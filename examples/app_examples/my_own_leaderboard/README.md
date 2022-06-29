# My Own LeaderBoard

This application is highly inspired from Kaggle internal workings and provide a leaderboard abstraction.
This could be used by a team of researchers to develop their models and keep track of the best one.
One could easily imagine how to extend this application to promote the best model to production.

# How does it work?

The application requires the user to select a `script_path`. When the user click submit,
this would queue the `work`. Once started, this would run the `script_path` into a subprocess and collect the `submission` file.
The submission file will be automatically compared to the ground truth and the associated metric is better, this run would be considered as a new entry of the leaderboard history.

# Which default data are provided?

This application relies on the [Tabular Playground Series - Jan 2022](https://www.kaggle.com/c/tabular-playground-series-jan-2022/data).
