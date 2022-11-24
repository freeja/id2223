**Intro**

The task of the lab was to build a serverless ML to with tools such as Hopsworks, Huggingface and Modal to predict if a randomly generated passenger would surive Titanic. The tasks were as follows:

1. Build and run a feature pipeline on Modal.
2. Build and run a training pipeline on Modal.
3. Build and run an inference pipeline with a Gradio UI on Huggingface spaces.


**Implementation**

First, the titanic-feature-pipeline was implemented where the titanic dataset was processed and generates the feature groups based on the data. This included dropping columns such as passengerid, name, cabin and ticket for their lack of predictive power.

The titanic-training-pipeline then takes the previously created feature groups and creates a feature view. The data is then split into train and test data (80% train, 20% test). The model used for the titanic dataset was the K-nearest-neighbor classifier to predict the outcome of titanic passengers. Using Huggingface and Gardio an interactive UI was then created to allow users to input values in the different feature groups and see the predicted outcome of "their" passenger. The titanic-feature-pipeline-daily is run to generate a new random passenger.

The titanic-batch-inference-pipeline predicts the outcome of the last generated passenger as well as previous predictions along with a confusion matrix of all the previous predictions.

**Huggingface spaces**

- Iris Huggingface links

https://huggingface.co/spaces/freeja/iris

https://huggingface.co/spaces/freeja/iris-monitor


- Titanic Huggingface links

https://huggingface.co/spaces/freeja/titanic

https://huggingface.co/spaces/freeja/titanic-monitor

