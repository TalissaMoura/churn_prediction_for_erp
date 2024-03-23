from typing import Union

import numpy as np
import pandas as pd


def convert_to_categoric(dataframe: pd.DataFrame, subset: Union[str, list]):
    """
    Converts column/columns in subset to category.

    Args:
        dataframe (pd.DataFrame): _description_
        subset (Union[str, list]): Column/Columns name to convert to category. If str
        then one column has type changed to float or if List, then each column in list
        has type convert to category.

    Returns:
        pd.DataFrame: A new DataFrame with the subset columns changed to type float.
    """   
    new_df = dataframe.copy()
    new_df[subset] = new_df[subset].astype("category")
    return new_df


def classify_col(
    dataframe: pd.DataFrame, col_to_clf: str, new_col_name: str, map: dict
) -> pd.DataFrame:
    """
    Creates a qualitative column from a quantitative column. Search
    for a range of values and sets a category for them.

    Example:

    >>>
    # Create a sample DataFrame
    df = pd.DataFrame({'A': ['R$ 1,000.00', 'R$ 2,500.50', 'R$ 3,750.75'],
                       'B': ['USD 100.00', 'USD 250.50', 'USD 375.75']})

    # Apply classify_col
    class_ranges = {"entre 1000 e 2800":range(1000,2800),"maior que 3000":range(3000,4000)}
    new_df = classify_col(dataframe=df,col_to_clf="A",new_col_name="clf_A",map=class_ranges)
    print(new_df)
    # Output:
    #         A            B        clf_A
    # 0  1000.00   USD 100.00    entre 1000 e 2800
    # 1  2500.50   USD 250.50    entre 1000 e 2800
    # 2  3750.75   USD 375.75    maior que 3000

    Args:
        dataframe (pd.DataFrame): input dataframe to add classify column.
        col_to_clf (str): quantitative column to classify.
        new_col_name (str): name of the new qualitative column.
        map (dict): a dict that contains the classes and range of values to
        classify columns. Example {"valores maiores que 10":range(10,20),"valor igual a 1 ou 2":[1,2]}

    Returns:
        pd.DataFrame: A new dataframe with new qualitative columns.
    """    
    new_df = dataframe.copy()
    new_df[new_col_name] = np.nan
    for cat, range in map.items():
        values = dataframe[col_to_clf].isin(range)
        new_df[new_col_name] = new_df[new_col_name].mask(values, cat)
    return new_df


def create_missing_indicator(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new columns that indicates a missing values in dataframe columns.
    Args:
        dataframe (pd.DataFrame): input DataFrame to search for missing values.

    Returns:
        pd.DataFrame: new DataFrame with missing value indicator columns. The new column names
        have name "is_columnname_null".
    """    
    new_df = dataframe.copy()
    is_null_in_cols = new_df.isnull().any()
    cols_with_null_values = list(is_null_in_cols[is_null_in_cols == True].index)
    for col in cols_with_null_values:
        new_df[f"is_{col}_null"] = new_df[col].isnull().astype(int)
    return new_df


def count_class_frequency(
    dataframe: pd.DataFrame, class_to_count: str, columns: Union[str, list]
) -> pd.DataFrame:
    """
    Count a frequency of a class on column/columns.

    Args:
        dataframe (pd.DataFrame): input DataFrame to count a class frequency.
        class_to_count (str): the class value to count in column/columns
        columns (Union[str, list]): column/columns to count a class frequency.
        if str count a frequency on one column or if list count a frequency of class
        in each column in list.

    Returns:
        pd.DataFrame: a new DataFrame with count frequency columns. The
        name of the columns have this pattern "qty_class_to_count".
    """    
    new_df = dataframe.copy()
    str_class_to_count = class_to_count.strip().lower().replace(" ", "")
    new_df[f"qty_{str_class_to_count}"] = (
        new_df[columns]
        .map(
            func=lambda s: 1 if s == class_to_count else 0,
        )
        .agg("sum", axis=1)
    )
    return new_df


def create_eq_or_gt_feature(
    dataframe: pd.DataFrame,
    value: Union[int, float],
    columns: Union[str, list],
    feature_name: str,
) -> pd.DataFrame:
    """
    Creates a new column in dataframe that check if a column/columns
    contains values bigger than other value

    Args:
        dataframe (pd.DataFrame): input DataFrame to create new feature.
        value (Union[int, float]): value to compare if the column/columns values are bigger or equal.
        columns (Union[str, list]): Column/columns to compare.
        feature_name (str): the name to apply to feature.

    Returns:
        pd.DataFrame: new DataFrame that contain the new feature.
    """    
    new_df = dataframe.copy()
    new_df[feature_name] = new_df[columns].gt(value) | new_df[columns].eq(value)
    new_df[feature_name] = new_df[feature_name].astype("int")
    return new_df
