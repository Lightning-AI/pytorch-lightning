import argparse

import numpy as np
import pandas

# Parse arguments provided by the Work.
parser = argparse.ArgumentParser()
parser.add_argument("--submission_path", type=str, required=True)
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
args = parser.parse_args()

# Read the provided data.
df_train = pandas.read_csv(args.train_data_path)
df_test = pandas.read_csv(args.test_data_path)

# Create random prediction.
min = df_train.num_sold.min()
max = df_train.num_sold.max()
df_test = pandas.read_csv(args.test_data_path)
submission_df = df_test.drop(["row_id", "date", "country", "store", "product"], axis=1)
submission_df.index.name = "row_id"
submission_df["num_sold"] = np.random.randint(min, max, len(df_test))

# Store submission
submission_df.to_csv(args.submission_path)
