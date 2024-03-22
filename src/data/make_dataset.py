# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Union

import click
import numpy as np
import pandas as pd

NUM_FEATS = ["receita_mensal", "receita_total"]


def drop_cols(dataframe: pd.DataFrame, subset=Union[str, list]) -> pd.DataFrame:
    new_df = dataframe.copy()
    new_df = new_df.drop(labels=subset, axis=1)
    return new_df


def rename_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
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
    dataframe: pd.DataFrame, subset=Union[str, list]
) -> pd.DataFrame:
    new_df = dataframe.copy()
    if isinstance(subset, str):
        new_df[subset] = (
            new_df[subset]
            .str.replace(pat=r"([R$])|([.])", repl="", regex=True)
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
    new_df = dataframe.copy()
    new_df[subset] = new_df[subset].astype("float")
    return new_df


def add_to_pipe(
    dataframe: pd.DataFrame, func=callable, *args, **kwargs
) -> pd.DataFrame.pipe:
    return dataframe.pipe(func, *args, **kwargs)


def make_pipeline(dataframe: pd.DataFrame, functions: list) -> pd.DataFrame:
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
