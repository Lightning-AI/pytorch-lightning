import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("Starting processing ...")
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(
    df_data.values, df_target.values, test_size=0.20, random_state=random.randint(0, 42)
)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Finished processing.")
