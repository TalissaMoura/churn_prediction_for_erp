import pandas as pd
import numpy as np
from typing import Union

def convert_to_categoric(dataframe:pd.DataFrame,subset:Union[str,list]):
    new_df = dataframe.copy()
    new_df[subset]=new_df[subset].astype("category")
    return new_df

def classify_col(dataframe:pd.DataFrame,col_to_clf:str,new_col_name:str,map:dict)->pd.DataFrame:
    new_df = dataframe.copy()
    new_df[new_col_name]=np.nan
    for cat,range in map.items():
        values = dataframe[col_to_clf].isin(range)
        new_df[new_col_name]=new_df[new_col_name].mask(values,cat)
    return new_df
def create_missing_indicator(dataframe:pd.DataFrame)->pd.DataFrame:
    new_df = dataframe.copy()
    is_null_in_cols = new_df.isnull().any()
    cols_with_null_values = list(is_null_in_cols[is_null_in_cols==True].index)
    for col in cols_with_null_values:
        new_df[f"is_{col}_null"] = new_df[col].isnull().astype(int)
    return new_df

def count_class_frequency(dataframe:pd.DataFrame,class_to_count:str,columns:Union[str,list])->pd.DataFrame:
    new_df = dataframe.copy()
    str_class_to_count = class_to_count.strip().lower().replace(" ","")
    new_df[f"qty_{str_class_to_count}"] = new_df[columns].map(
            func=lambda s: 1 if s == class_to_count else 0,
        ).agg("sum",axis=1)
    return new_df

def create_eq_or_gt_feature(dataframe:pd.DataFrame,value:Union[int,float],columns:Union[str,list],feature_name:str)->pd.DataFrame:
    new_df = dataframe.copy()
    new_df[feature_name] = new_df[columns].gt(value)|new_df[columns].eq(value)
    new_df[feature_name] = new_df[feature_name].astype("int")
    return new_df