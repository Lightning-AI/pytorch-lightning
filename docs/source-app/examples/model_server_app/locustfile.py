from locust import FastHttpUser, task
from sklearn import datasets
from sklearn.model_selection import train_test_split


class HelloWorldUser(FastHttpUser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_inference_request()

    @task
    def predict(self):
        self.client.post(
            "/v2/models/mnist-svm/versions/v0.0.1/infer",
            json=self.inference_request,
        )

    def _prepare_inference_request(self):
        # The digits dataset
        digits = datasets.load_digits()

        # To apply a classifier on this data,
        # we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        # Split data into train and test subsets
        _, X_test, _, _ = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

        x_0 = X_test[0:1]
        self.inference_request = {
            "inputs": [
                {
                    "name": "predict",
                    "shape": x_0.shape,
                    "datatype": "FP32",
                    "data": x_0.tolist(),
                }
            ]
        }
