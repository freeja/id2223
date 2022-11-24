import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
titanic_df.dropna(inplace=True) #drop rows with missing values
titanic_df.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
titanic_df.Embarked.replace({'S':1, 'C':2, 'Q':3}, inplace=True)
titanic_df.Sex.replace({'male':0, 'female':1}, inplace=True)


titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["pclass","sex","age","sibsp","parch","fare","embarked"],
    description="Titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job": False})
