import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(survived, pclass_min, pclass_max, 
                    age_max, age_min, sibsp_min, sibsp_max,parch_min,parch_max,fare_max,fare_min,embarked_min,
                     embarked_max):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    #survived = random.randint(0,1)

    passenger_df = pd.DataFrame({ "pclass": [random.randint(pclass_min, pclass_max)],
                       "sex": [random.randint(0, 1)],
                       "age": [random.uniform(age_min, age_max)],
                       "sibsp": [random.randint(sibsp_min, sibsp_max)],
                       "parch":[random.randint(parch_min,parch_max)],
                       "embarked":[random.randint(embarked_min,embarked_max)],
                       "fare":[random.uniform(fare_min,fare_max)]
                      })
    passenger_df['survived'] = survived
    return passenger_df


def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random Titanic passenger
    """
    import pandas as pd
    import random

    survived_df = generate_passenger(1,0,3,80,0.42,0,3,0,2,512.3292,0,1,3)
    dead_df = generate_passenger(0,0,3,74,1,0,3,0,4,227.535,0,1,3)
    # randomly pick one of these  and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = survived_df
        print("Survived added")
    else:
        passenger_df = dead_df
        print("Dead added")

    return passenger_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    passenger_df = get_random_titanic_passenger()

    passenger_fg = fs.get_feature_group(name="titanic_modal",version=1)
    passenger_fg.insert(passenger_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
