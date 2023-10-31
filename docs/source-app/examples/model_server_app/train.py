import joblib
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

from lightning import LightningWork
from lightning.app.storage.path import Path


class TrainModel(LightningWork):
    """This component trains a Sklearn SVC model on digits dataset."""

    def __init__(self):
        super().__init__()
        # 1: Add element to the state.
        self.best_model_path = None

    def run(self):
        # 2: Load the Digits
        digits = datasets.load_digits()

        # 3: To apply a classifier on this data,
        # we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        # 4: Create a classifier: a support vector classifier
        classifier = svm.SVC(gamma=0.001)

        # 5: Split data into train and test subsets
        X_train, _, y_train, _ = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

        # 6: We learn the digits on the first half of the digits
        classifier.fit(X_train, y_train)

        # 7: Save the Sklearn model with `joblib`.
        model_file_name = "mnist-svm.joblib"
        joblib.dump(classifier, model_file_name)

        # 8: Keep a reference the the generated model.
        self.best_model_path = Path("mnist-svm.joblib")
