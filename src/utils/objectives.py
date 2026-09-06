import os
import warnings

import optuna
import pandas as pd
import numpy as np


import xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    root_mean_squared_error,
    precision_recall_curve,
)
from sklearn.ensemble import IsolationForest


def _resolve_xgb_device() -> str:
    """Which device XGBoost should train on: "cuda" when one is usable, else "cpu".

    The measured runs behind the article all ran on a GTX 1650 Max-Q, so "cuda" stays the
    answer on any machine that has a working CUDA build -- the stored notebook outputs are
    reproduced unchanged there. On a CPU-only machine the notebooks used to die on the
    first Optuna trial; they now fall back instead. Set CAA_XGB_DEVICE to force either.
    """
    override = os.environ.get("CAA_XGB_DEVICE")
    if override:
        return override

    if not xgboost.build_info().get("USE_CUDA", False):
        return "cpu"

    # A CUDA-enabled build still needs a visible device at run time; the cheapest honest
    # test is a one-tree fit, which raises rather than warns from XGBoost 2.1 on.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            XGBClassifier(device="cuda", n_estimators=1, verbosity=0).fit(
                np.zeros((4, 1)), np.array([0, 1, 0, 1])
            )
        return "cuda"
    except Exception:  # pylint: disable=broad-except
        return "cpu"


XGB_DEVICE = _resolve_xgb_device()


XGB_PARAMS_RANGES = {
    "n_estimators": (100, 600),
    "learning_rate": (1e-4, 1),
    "max_depth": (1, 10),
    "min_child_weight": (1, 20),
    "subsample": (0.2, 1),
    "colsample_bytree": (0.2, 1),
    "colsample_bylevel": (0.2, 1),
    "gamma": (0, 10),
    "alpha": (0, 10),
    "lambda": (0, 10),
}

ISOFOREST_PARAMS_RANGES = {
    "n_estimators": (50, 300),
    "contamination": (0.01, 0.1),
    "max_samples": (0.5, 1.0),
    "max_features": (0.5, 1.0),
}


def get_sorted_features(model, best_features):
    return [
        str(k)
        for k in np.array(best_features)[model.feature_importances_.argsort()][::-1]
    ]


def get_close_ranges(params: dict, rate: float = 0.1) -> dict:

    close_range = {}
    for param, value in params.items():
        if isinstance(value, int):
            delta = max(int(value * rate), 1)
            close_range[param] = (value - delta, value + delta)
        elif isinstance(value, float):
            delta = value * rate
            if "sample" in param:
                close_range[param] = (max(0.0, value - delta), min(value + delta, 1.0))
            else:
                close_range[param] = (max(0.0, value - delta), value + delta)
        elif isinstance(value, bool):
            close_range[param] = (True, False)
        else:
            pass
    return close_range


def get_best_isolation_model(
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
) -> XGBClassifier:
    """Get the best multiclass model"""
    print("used features:", list(x_train.columns))
    # Train Isolation Forest
    model = IsolationForest(
        random_state=42,
        **best_params,
    )
    # training on normal data only
    model.fit(x_train[y_train == 0])

    # Evaluate using Average Precision Score
    print("F1-Score Train:", get_f1_score(y_train, x_train, model))
    print("F1-Score Dev:  ", get_f1_score(y_dev, x_dev, model))
    return model


def get_f1_score(y_true, x_dev, model):
    """Get the F1 score for the given model"""
    # Predict anomaly scores
    anomaly_scores = model.decision_function(x_dev)  # Higher means more anomalous

    # computing precision and recall
    precisions, recalls, _ = precision_recall_curve(y_true, anomaly_scores)

    # Compute F1-score for each threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)

    return max(f1_scores)


# Optuna Objective Function
def get_isolation_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    custom_ranges: dict = None,
):
    """Isolation forest objective function for Optuna"""

    ranges = ISOFOREST_PARAMS_RANGES
    if custom_ranges is not None:
        ranges.update(custom_ranges)

    def objective(trial: optuna.Trial) -> float:
        """The objective function to minimize"""

        trial.suggest_int("n_estimators", *ranges.get("n_estimators"))
        trial.suggest_float("contamination", *ranges.get("contamination"))
        trial.suggest_float("max_samples", *ranges.get("max_samples"))
        trial.suggest_float("max_features", *ranges.get("max_features"))

        # Train Isolation Forest
        model = IsolationForest(
            random_state=42,
            **trial.params,
        )
        # training on normal data only
        model.fit(x_train[y_train == 0])

        # Evaluate using Average Precision Score
        return -get_f1_score(y_dev, x_dev, model)

    return objective


def get_multiclass_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    objective: str = "multi:softprob",
    w_train: pd.DataFrame = None,
    w_dev: pd.DataFrame = None,
    custom_ranges: dict = None,
    # select_features: bool = False,
    sorted_features: list = None,
    max_features_removed: int = 20,
):
    """Multiclass objective function for Optuna"""

    ranges = XGB_PARAMS_RANGES
    if custom_ranges is not None:
        ranges.update(custom_ranges)

    feature_names = x_train.columns.tolist()  # Get all feature names

    def objective_function(trial: optuna.Trial) -> float:
        """The objective function to minimize"""

        # setting model parameters
        trial.suggest_categorical("objective", [objective])
        trial.suggest_int("n_estimators", *ranges.get("n_estimators"))
        trial.suggest_float("learning_rate", *ranges.get("learning_rate"))
        trial.suggest_int("max_depth", *ranges.get("max_depth"))
        trial.suggest_int("min_child_weight", *ranges.get("min_child_weight"))
        trial.suggest_float("subsample", *ranges.get("subsample"))
        trial.suggest_float("colsample_bytree", *ranges.get("colsample_bytree"))
        trial.suggest_float("colsample_bylevel", *ranges.get("colsample_bylevel"))
        trial.suggest_float("gamma", *ranges.get("gamma"))
        trial.suggest_float("alpha", *ranges.get("alpha"))
        trial.suggest_float("lambda", *ranges.get("lambda"))

        # inititating model with paramters
        model = XGBClassifier(
            num_class=y_train.nunique(),
            device=XGB_DEVICE,
            n_jobs=-1,
            random_state=42,
            **trial.params,
        )

        # Feature selection: Let Optuna decide whether to keep each feature
        selected_features = feature_names
        if sorted_features is not None:
            trial.suggest_int("n_features_removed", 0, max_features_removed)
            selected_features = [
                f
                for f in feature_names
                if f
                in sorted_features[
                    : len(sorted_features) - trial.params["n_features_removed"]
                ]
            ]
            # selected_features = [
            #     feature
            #     for feature in feature_names
            #     if trial.suggest_categorical(f"keep_{feature}", [True, False])
            # ]

        # Subset the dataset with selected features
        x_train_ = x_train[selected_features]
        x_dev_ = x_dev[selected_features]

        # fitting on sample
        model.fit(x_train_, y_train, sample_weight=w_train)

        # getting perf on dev
        preds = model.predict_proba(x_dev_)
        return log_loss(y_dev, preds, sample_weight=w_dev)

    return objective_function


def get_best_multiclass_model(
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    w_train: pd.DataFrame = None,
    w_dev: pd.DataFrame = None,
    train_on_full: bool = True,
) -> XGBClassifier:
    """Get the best multiclass model"""

    # Identify selected features based on best_params
    selected_features = [
        col for col in x_train.columns if best_params.get(f"keep_{col}", True)
    ]

    # Filter dataset to use only selected features
    x_train = x_train[selected_features]
    x_dev = x_dev[selected_features]

    print("used features:", list(x_train.columns))

    # Train the model on the selected features
    model = XGBClassifier(
        num_class=y_train.nunique(),
        device=XGB_DEVICE,
        n_jobs=-1,
        random_state=42,
        **{k: v for k, v in best_params.items() if not k.startswith("keep_")},
    )
    model = model.fit(x_train, y_train, sample_weight=w_train)

    # evaluating performances
    print(
        "Log Loss Train:",
        log_loss(y_train, model.predict_proba(x_train), sample_weight=w_train),
    )
    print(
        "Log Loss Dev:  ",
        log_loss(y_dev, model.predict_proba(x_dev), sample_weight=w_dev),
    )

    # training on complete dataset
    if train_on_full:
        x = pd.concat([x_train, x_dev])
        y = pd.concat([y_train, y_dev])

        w = None
        if w_train is not None and w_dev is not None:
            w = pd.concat([pd.Series(w_train), pd.Series(w_dev)])
            w.index = x.index

        model = model.fit(x, y, sample_weight=w)
        print(
            "Log Loss Overall:  ", log_loss(y, model.predict_proba(x), sample_weight=w)
        )
    return selected_features, model


def get_binary_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    custom_ranges: dict = None,
):
    """binary objective function for optuna"""

    ranges = XGB_PARAMS_RANGES
    if custom_ranges is not None:
        ranges.update(custom_ranges)

    def objective(trial: optuna.Trial) -> float:
        """the objective function to minimize"""

        trial.suggest_int("n_estimators", *ranges.get("n_estimators"))
        trial.suggest_float("learning_rate", *ranges.get("learning_rate"))
        trial.suggest_int("max_depth", *ranges.get("max_depth"))
        trial.suggest_int("min_child_weight", *ranges.get("min_child_weight"))
        trial.suggest_float("subsample", *ranges.get("subsample"))
        trial.suggest_float("colsample_bytree", *ranges.get("colsample_bytree"))
        trial.suggest_float("colsample_bylevel", *ranges.get("colsample_bylevel"))
        trial.suggest_float("gamma", *ranges.get("gamma"))
        trial.suggest_float("alpha", *ranges.get("alpha"))
        trial.suggest_float("lambda", *ranges.get("lambda"))

        model = XGBClassifier(
            objective="binary:logistic",
            device=XGB_DEVICE,
            n_jobs=-1,
            random_state=42,
            **trial.params,
        )
        model.fit(x_train, y_train)
        preds = model.predict_proba(x_dev)[:, 1]
        return -roc_auc_score(y_dev, preds)

    return objective


def get_best_binary_model(
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
) -> XGBClassifier:
    """Get the best binary model"""
    print("used features:", list(x_train.columns))
    model = XGBClassifier(
        objective="binary:logistic",
        device=XGB_DEVICE,
        n_jobs=-1,
        random_state=42,
        **best_params,
    )
    model = model.fit(x_train, y_train)
    print("AUC Train:", roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]))
    print("AUC Dev:  ", roc_auc_score(y_dev, model.predict_proba(x_dev)[:, 1]))

    x = pd.concat([x_train, x_dev])
    y = pd.concat([y_train, y_dev])
    model = model.fit(x, y)
    print("AUC Overall:  ", roc_auc_score(y, model.predict_proba(x)))
    return model


def get_regression_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    objective: str = "reg:squarederror",
    w_train: pd.DataFrame = None,
    w_dev: pd.DataFrame = None,
    custom_ranges: dict = None,
    # select_features: bool = False,
    sorted_features: list = None,
):
    """regression objective function for optuna"""

    ranges = XGB_PARAMS_RANGES
    if custom_ranges is not None:
        ranges.update(custom_ranges)

    feature_names = x_train.columns.tolist()  # Get all feature names

    def objective_function(trial: optuna.Trial) -> float:
        """the objective function to minimize"""

        # setting model parameters
        trial.suggest_categorical("objective", [objective])
        trial.suggest_int("n_estimators", *ranges.get("n_estimators"))
        trial.suggest_float("learning_rate", *ranges.get("learning_rate"))
        trial.suggest_int("max_depth", *ranges.get("max_depth"))
        trial.suggest_int("min_child_weight", *ranges.get("min_child_weight"))
        trial.suggest_float("subsample", *ranges.get("subsample"))
        trial.suggest_float("colsample_bytree", *ranges.get("colsample_bytree"))
        trial.suggest_float("colsample_bylevel", *ranges.get("colsample_bylevel"))
        trial.suggest_float("gamma", *ranges.get("gamma"))
        trial.suggest_float("alpha", *ranges.get("alpha"))
        trial.suggest_float("lambda", *ranges.get("lambda"))
        if objective == "reg:tweedie":
            trial.suggest_float("tweedie_variance_power", 1.2, 2.0)

        # inititating model with paramters
        model = XGBRegressor(
            # objective="reg:squarederror",
            device=XGB_DEVICE,
            n_jobs=-1,
            random_state=42,
            **trial.params,
        )

        # Feature selection: Let Optuna decide whether to keep each feature
        selected_features = feature_names  # Default to all features
        if sorted_features is not None:
            trial.suggest_int("n_features_removed", 0, len(sorted_features) - 20)
            selected_features = [
                f
                for f in feature_names
                if f
                in sorted_features[
                    : len(sorted_features) - trial.params["n_features_removed"]
                ]
            ]
        # if select_features:
        #     selected_features = [
        #         feature
        #         for feature in feature_names
        #         if trial.suggest_categorical(f"keep_{feature}", [True, False])
        #     ]

        # Subset the dataset with selected features
        x_train_ = x_train[selected_features]
        x_dev_ = x_dev[selected_features]

        # fitting on sample
        model.fit(x_train_, y_train, sample_weight=w_train)

        # getting perf on dev
        pred = model.predict(x_dev_)

        # `reg:tweedie` with `tweedie_variance_power` near 2 and an aggressive learning
        # rate can diverge to non-finite predictions; scoring those raises instead of
        # returning a value, which kills the whole study. Such a model is unusable, so
        # score it +inf and let the sampler steer away from that region.
        if not np.isfinite(pred).all():
            return float("inf")

        return root_mean_squared_error(
            y_dev,
            pred,  # sample_weight=w_dev
        )

    return objective_function


def get_best_regression_model(
    best_params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    w_train: pd.DataFrame = None,
    w_dev: pd.DataFrame = None,
    train_on_full: bool = True,
) -> XGBClassifier:
    """Get the best binary model"""

    # Identify selected features based on best_params
    selected_features = [
        col for col in x_train.columns if best_params.get(f"keep_{col}", True)
    ]

    # Filter dataset to use only selected features
    x_train = x_train[selected_features]
    x_dev = x_dev[selected_features]

    print("used features:", list(x_train.columns))

    # Train the model on the selected features
    model = XGBRegressor(
        device=XGB_DEVICE,
        n_jobs=-1,
        random_state=42,
        **{k: v for k, v in best_params.items() if not k.startswith("keep_")},
    )
    model = model.fit(x_train, y_train, sample_weight=w_train)

    # evaluating performances
    print(
        "RMSE Train:",
        root_mean_squared_error(
            y_train,
            model.predict(x_train),  # sample_weight=w_train
        ),
    )
    print(
        "RMSE Dev:  ",
        root_mean_squared_error(
            y_dev,
            model.predict(x_dev),  # sample_weight=w_dev
        ),
    )

    # training on complete dataset
    if train_on_full:
        x = pd.concat([x_train, x_dev])
        y = pd.concat([y_train, y_dev])
        w = None
        if w_train is not None and w_dev is not None:
            w = pd.concat([pd.Series(w_train), pd.Series(w_dev)])
            w.index = x.index

        model = model.fit(x, y, sample_weight=w)
        print(
            "RMSE Overall:  ",
            root_mean_squared_error(
                y,
                model.predict(x),  # sample_weight=w
            ),
        )
    return selected_features, model


# def get_poisson_objective(
#     x_train: pd.DataFrame,
#     y_train: pd.Series,
#     x_dev: pd.DataFrame,
#     y_dev: pd.DataFrame,
#     w_train: pd.DataFrame = None,
#     w_dev: pd.DataFrame = None,
#     custom_ranges: dict = None,
# ):
#     """poisson regression objective function for optuna"""

#     ranges = XGB_PARAMS_RANGES
#     if custom_ranges is not None:
#         ranges.update(custom_ranges)

#     def objective(trial: optuna.Trial) -> float:
#         """the objective function to minimize"""

#         trial.suggest_int("n_estimators", *ranges.get("n_estimators"))
#         trial.suggest_float("learning_rate", *ranges.get("learning_rate"))
#         trial.suggest_int("max_depth", *ranges.get("max_depth"))
#         trial.suggest_int("min_child_weight", *ranges.get("min_child_weight"))
#         trial.suggest_float("subsample", *ranges.get("subsample"))
#         trial.suggest_float("colsample_bytree", *ranges.get("colsample_bytree"))
#         trial.suggest_float("colsample_bylevel", *ranges.get("colsample_bylevel"))
#         trial.suggest_float("gamma", *ranges.get("gamma"))
#         trial.suggest_float("alpha", *ranges.get("alpha"))
#         trial.suggest_float("lambda", *ranges.get("lambda"))

#         model = XGBRegressor(
#             objective="count:poisson",
#             device="cuda",
#             n_jobs=-1,
#             random_state=42,
#             **trial.params,
#         )
#         model.fit(x_train, y_train, sample_weight=w_train)

#         return root_mean_squared_error(y_dev, model.predict(x_dev), sample_weight=w_dev)

#     return objective


# def get_best_poisson_model(
#     best_params: dict,
#     x_train: pd.DataFrame,
#     y_train: pd.Series,
#     x_dev: pd.DataFrame,
#     y_dev: pd.DataFrame,
#     w_train: pd.DataFrame = None,
#     w_dev: pd.DataFrame = None,
# ) -> XGBClassifier:
#     """Get the best poisson model"""
#     print("used features:", list(x_train.columns))
#     model = XGBRegressor(
#         objective="count:poisson",
#         device="cuda",
#         n_jobs=-1,
#         random_state=42,
#         **best_params,
#     )
#     model = model.fit(x_train, y_train, sample_weight=w_train)
#     print(
#         "RMSE Train:",
#         root_mean_squared_error(y_train, model.predict(x_train), sample_weight=w_train),
#     )
#     print("RMSE Dev:  ", root_mean_squared_error(y_dev, model.predict(x_dev), sample_weight=w_dev))

#     x = pd.concat([x_train, x_dev])
#     y = pd.concat([y_train, y_dev])
#     w = None
#     if w_train is not None and w_dev is not None:
#         w = pd.concat([pd.Series(w_train), pd.Series(w_dev)])
#         w.index = x.index
#     model = model.fit(x, y, sample_weight=w)
#     print("RMSE Overall:  ", root_mean_squared_error(y, model.predict(x), sample_weight=w))
#     return model
