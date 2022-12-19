
*********
Objective
*********

Learn how to train Scikit-learn models using Lightning AI with large-scale datasets.

Lightning AI manages all cloud infrastructure provisioning for us while we focus on machine learning.
If you are new to the Lightning AI latest framework learn more about it in
this blog `How to Build a Machine Learning Training and Deployment Pipeline <https://lightning.ai/pages/community/tutorial/ml-training-deployment/>`_.

----

*********************************************
Building a Scikit-learn training component
*********************************************

We will be using the Iris flower dataset composed of three kinds of irises having different sepal and petal lengths.
The below chart shows the distribution of flowers by their sepal width and length.


.. figure::  https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dataset_002.png


Luckily, Scikit learn has a prebuilt function to load the iris dataset.
We will load the data using scikit learn and train a Decision Tree classifier for this tutorial.

.. literalinclude:: ./train.py

----

Now, we will create a Lightning component that will run our training job. To run a long-running task like downloading a dataset and training a machine learning model we use LightningWork.

We create a class SKLearnTraining that inherits LightningWork and defines the run method where we will run all the steps for training our model. We can also configure the machine-related settings like CPU, RAM, and disk size using the CloudCompute API.

.. literalinclude:: ./app.py

In order to save the model for deployment or fine-tuning in the future we need to save the model. Lightning provides Drive API, a central place for components to share data.
We can store our model in the drive and easily access it either from a different component in our workflow like a deployment pipeline or download the model manually from the Lightning App dashboard.

To run this app, we will create an app object using LigthningApp and save it to app.py module.

Finally, to run this training process on the Lightning cloud platform run lightning run app app.py --cloud. You can drop the --cloud flag to run this locally on your own machine.

----

*********************************************************
Migrate existing Machine Learning training to Lightning
*********************************************************

If you already have a scikit-learn training process on your local system or some other cloud provided,
all you need to do is create a LightningWork class and define the run method.

You can define all the configs in the `__init__` method and move the training code to the run method.
The run method is the place where Lightning will execute your training process.
You can sign up to a Lightning AI account to get 3 free credits and try your application on the cloud using the lightning run app app.py --cloud command.

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Deploy a model
   :description: Learn how to deploy a model
   :col_css: col-md-4
   :button_link: model_server_app/model_server_app.html
   :height: 180
   :tag: intermediate
