import lightning as L
from lightning.app.storage import Drive


from sklearn.datasets import load_iris
from sklearn import tree
from joblib import dump, load

class SKLearnTraining(L.LightningWork):
    def __init__(self):
        # configure the machine related config using CloudCompute API
        super().__init__(cloud_compute=L.CloudCompute("cpu", disk_size=10))

        # cloud persistable storage for model checkpoint
        self.model_storage = Drive("lit://checkpoints")

    def run(self):
        # Step 1
        # Download the dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Step 2
        # Intialize the model
        clf = tree.DecisionTreeClassifier()

        # Step 3
        # Train the model
        clf = clf.fit(X, y)

        # Step 4
        # Save the model
        dump(clf, 'model.joblib')

        self.model_storage.put("model.joblib")
        print("model trained and saved successfully")

component = SKLearnTraining()
app = L.LightningApp(component)
