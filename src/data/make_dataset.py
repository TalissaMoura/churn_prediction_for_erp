# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Union

import click
import numpy as np
import pandas as pd

NUM_FEATS = ["receita_mensal", "receita_total"]


def drop_cols(dataframe: pd.DataFrame, subset:Union[str, list]) -> pd.DataFrame:
    """
    Remove columns from DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame to drop columns.
        subset (Union[str, list]): Column/Columns names to drop

    Returns:
        pd.DataFrame: a new DataFrame with dropped columns.
    """    
    new_df = dataframe.copy()
    new_df = new_df.drop(labels=subset, axis=1)
    return new_df


def rename_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and renames its columns based on the following steps:
    1. Strips leading and trailing whitespaces from column names.
    2. Converts column names to lowercase.
    3. Replaces spaces in column names with underscores.
    4. Removes special characters (":", ".", and ",") from column names.

    At the end a column named "Column:A " has changed to "column_A".

    Args:
        dataframe (pd.DataFrame): The input DataFrame whose columns are to be renamed.

    Returns:
        pd.DataFrame: A new DataFrame with the columns renamed according to the described steps.
    """    
    new_df = dataframe.copy()
    new_cols_name = (
        pd.Series(dataframe.columns)
        .str.strip()
        .str.lower()
        .str.replace(pat=" ", repl="_")
        .str.replace(pat=r"([:.,])", repl="", regex=True)
        .astype("string")
    )
    new_cols_name.index = dataframe.columns
    new_df = new_df.rename(mapper=new_cols_name.to_dict(), axis=1)
    return new_df


def clear_numeric_strings(
    dataframe: pd.DataFrame, subset:Union[str, list]
) -> pd.DataFrame:
    """
    This function clear numeric string that comes like this pattern: "R$ 76,86 " and changes
    to "76.86". For each column in the subset, these steps are applied:
    1. Remove "R$". 
    2. Strips leading and trailing whitespaces from column names.
    3. Replace "," to "." 

    Args:
        dataframe (pd.DataFrame): Input dataframe to clear strings.
        subset (Union[str, list]): Column/Columns names to clear the numeric strings. If str then changes are applied in one column or if list
        then changes are applied in each column in the list.

    Returns:
        pd.DataFrame: A new DataFrame with numeric strings cleaned in the specified columns.
    """    
    new_df = dataframe.copy()
    if isinstance(subset, str):
        new_df[subset] = (
            new_df[subset]
            .str.replace(pat=r"([R$])", repl="", regex=True)
            .str.strip()
            .str.replace(pat=",", repl=".")
        )
    else:
        new_df[subset] = new_df[subset].apply(
            func=lambda s: s.str.replace(pat=r"([R$])|([.])", repl="", regex=True)
            .str.strip()
            .str.replace(pat=",", repl="."),
            axis=1,
        )
    return new_df


def convert_to_numeric(
    dataframe: pd.DataFrame, subset: Union[str, list]
) -> pd.DataFrame:
    """
    Converts column/columns in subset to float64.

    Args:
        dataframe (pd.DataFrame): _description_
        subset (Union[str, list]): Column/Columns name to convert to float. If str
        then one column has type changed to float or if List, then each column in list
        has type convert to float64.

    Returns:
        pd.DataFrame: A new DataFrame with the subset columns changed to type float.
    """    
    new_df = dataframe.copy()
    new_df[subset] = new_df[subset].astype("float")
    return new_df


def add_to_pipe(
    dataframe: pd.DataFrame, func:callable, *args, **kwargs
) -> pd.DataFrame.pipe:
    """
    Add a function to be applied on dataframe to a pipeline using
    the Pandas.DataFrame.pipe function. 

    Args:
        dataframe (pd.DataFrame): Input DataFrame to apply the function to
        func (callable): A function be applied on dataframe.
        *args: Positional arguments to apply on function.
        **kwargs: Keyword arguments to apply on function.

    Returns:
        pd.DataFrame.pipe: a pipeline with function to apply on input dataframe
    """    
    return dataframe.pipe(func, *args, **kwargs)


def make_pipeline(dataframe: pd.DataFrame, functions: list[dict]) -> pd.DataFrame:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_
        functions (list): List of dict with functions to apply on input dataframe.
        The dicts have this pattern:
        {"function":func,"function_kwargs":{"keyword_1":value},"function_args":[value1,value2]}
        The dict must have a function but function_kwargs and function_args can be optional.
        If function_kwargs in dict its value must be a dict and if function_args in 
        dict then its must be a iterable.

    Returns:
        pd.DataFrame: A new DataFrame with all the functions listed applied on input dataframe
    """    

    new_df = dataframe.copy()
    for f in functions:
        if "function_args" in f.keys() and "function_kwargs" in f.keys():
            try:
                new_df = add_to_pipe(
                    new_df,
                    *f["function_args"],
                    func=f["function"],
                    **f["function_kwargs"],
                )
            except TypeError:
                print(
                    "function_args must be a iterable and function_kwargs must be a dict."
                )
        elif "function_kwargs" in f.keys() and "function_args" not in f.keys():
            try:
                new_df = add_to_pipe(new_df, func=f["function"], **f["function_kwargs"])
            except TypeError:
                raise Exception(f"function_kwargs must be a dict")
        elif "function_args" in f.keys() and "function_kwargs" not in f.keys():
            try:
                new_df = add_to_pipe(new_df, *f["function_args"], func=f["function"])
            except TypeError:
                raise Exception(f"function must be a iterable")
        else:
            try:
                new_df = add_to_pipe(new_df, func=f["function"])
            except TypeError:
                raise Exception(f"function must be a callable")
            except KeyError:
                raise Exception(f"function must be a key on dict.")
    return new_df


list_funcs = [
    {"function": drop_cols, "function_kwargs": {"subset": ["Emite boletos.1", "ID"]}},
    {
        "function": rename_cols,
    },
    {"function": clear_numeric_strings, "function_kwargs": {"subset": NUM_FEATS}},
    {"function": convert_to_numeric, "function_kwargs": {"subset": NUM_FEATS}},
]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    raw_df = pd.read_csv(input_path / "customer_churn_data - customer_churn_data.csv")

    cleared_df = make_pipeline(dataframe=raw_df, functions=list_funcs)

    cleared_df.to_csv(output_path / "cleared_df.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
