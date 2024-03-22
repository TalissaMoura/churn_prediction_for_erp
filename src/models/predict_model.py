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
