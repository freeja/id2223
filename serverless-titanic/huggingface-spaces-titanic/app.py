import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def iris(pclass, sex, age, sibsp,parch,fare,embarked):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(fare)
    input_list.append(embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    survival_url = "https://raw.githubusercontent.com/freeja/id2223/main/serverless-titanic/" + str(res[0]) + ".png"
    img = Image.open(requests.get(survival_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=iris,
    title="Titanic Survival Analytics",
    description="Experiment with passenger stats to predict the outcome of their journey.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="pclass (1-3)"),
        gr.inputs.Number(default=1.0, label="sex (male:0, female:1)"),
        gr.inputs.Number(default=1.0, label="age (1-80)"),
        gr.inputs.Number(default=1.0, label="sibsp (0-3)"),
        gr.inputs.Number(default=1.0, label="parch (0-4)"),
        gr.inputs.Number(default=1.0, label="fare (0-512)"),
        gr.inputs.Number(default=1.0, label="embarked (1-3)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

