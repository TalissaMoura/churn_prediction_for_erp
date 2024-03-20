import yaml
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from typing import Union
from pathlib import Path


def make_train(X_train:Union[np.array,pd.DataFrame],
               y_train:Union[np.array,pd.DataFrame],
               config:Path):
    
    with open(config,"r") as cfg_file:
        cfg=yaml.safe_load(cfg_file)

    num_features = [col.decode(encoding="ISO-8859-1") for col in cfg["model_features"]["NUM_FEATURES"]]
    cat_features = [col.decode(encoding="ISO-8859-1") for col in cfg["model_features"]["CAT_FEATURES"]]

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', OneHotEncoder(), cat_features)
        ],
    )
    # Define the pipeline
    model_params = cfg["model_parameters"]
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing (imputation + one-hot encoding)
    ('model', RandomForestClassifier(**model_params["fit_params"]))  # Model
    ])
    return pipeline.fit(X_train,y_train)