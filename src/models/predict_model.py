from typing import Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def make_predict(
    model: Pipeline,
    X_test: Union[pd.DataFrame, np.array],
    threshold: float = 0.5,
    use_predict_proba: Union[True, False] = True,
    return_classes: Union[False, True] = False,
):
    """
    Predict values for model in X_test. Calculates the predict_proba
    and predict_class. If a different value of threshold is set than
    predict_classes will be from predict_proba (predict_proba > threshold).

    Args:
        model (Pipeline): The model pipepline fited to make predict.
        X_test (Union[pd.DataFrame, np.array]): Test pd.DataFrame or np.array to apply the model
        threshold (float, optional): Threshold to make the decision to churn . Defaults to 0.5.
        use_predict_proba (Union[True, False], optional): Return the predicted probabilities . Defaults to True.
        return_classes (Union[False, True], optional): Return the predicted classes. Defaults to False.

    Returns:
        y_pred_proba,y_pred: predicted values, can return the predict_proba and
        predict classes or only predict_proba or only predict_classes.
    """

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    if threshold == 0.5:
        y_pred_cls = model.predict(X_test)
    else:
        y_pred_cls = np.where(y_pred_proba > threshold, 1, 0)

    if use_predict_proba and return_classes:
        return y_pred_proba, y_pred_cls

    elif not use_predict_proba and return_classes:
        return y_pred_cls
    else:
        return y_pred_proba
