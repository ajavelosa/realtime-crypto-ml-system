"""
Machine Learning Models with Hyperparameter Tuning

This module provides a framework for creating machine learning models with automatic
hyperparameter tuning using Optuna for Bayesian optimization.

USAGE:
    # Create model instance
    model = ModelWithHyperparameterTuning('RandomForestRegressor')

    # Fit with hyperparameter tuning (100 trials, 3 CV splits)
    model.fit(X_train, y_train, hyperparam_search_trials=100, hyperparam_splits=3)

    # Fit without hyperparameter tuning (uses default parameters)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

AVAILABLE MODELS:
    All models are now configured through model_hyperparameters.json:
    - LinearRegression
    - SGDRegressor
    - HuberRegressor
    - OrthogonalMatchingPursuit
    - LarsCV
    - RandomForestRegressor
    - LassoCV
    - PassiveAggressiveRegressor
"""

import importlib
import json
import os
from typing import Optional

import mlflow
import numpy as np
import optuna
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _get_class_from_string(class_path: str) -> type:
    """
    Dynamically import a class from a string path.

    Args:
        class_path: String path to the class (e.g., 'sklearn.linear_model.LinearRegression')

    Returns:
        The class object
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_model_configs() -> dict:
    """
    Load model configurations from the JSON file.

    Returns:
        Dictionary containing model configurations
    """
    config_path = os.path.join(os.path.dirname(__file__), 'model_hyperparameters.json')
    with open(config_path, 'r') as f:
        return json.load(f)


class BaselineModel:
    def __init__(self):
        """
        Initialize the baseline model.

        Args:
            X: The features.
            y: The target.
        """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the baseline model. We will pass since the baseline model
        will not be based on machine learning.
        """
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target using the baseline model. The baseline model
        will predict the future close price as the current close price.

        Args:
            X: The features.

        Returns:
            The predicted target.
        """
        return pd.Series(X['close'])


class ModelWithHyperparameterTuning:
    """
    Base class for models with hyperparameter tuning.
    Loads configuration from JSON file based on model name.
    """

    def __init__(self, model_name: str):
        """
        Initialize the model with configuration loaded from JSON.

        Args:
            model_name: Name of the model to load configuration for
        """
        self.model_name = model_name
        self.config = _load_model_configs()

        if model_name not in self.config['models']:
            available_models = list(self.config['models'].keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")

        model_config = self.config['models'][model_name]
        self.model_class = _get_class_from_string(model_config['model_class'])
        self.use_scaler = model_config['use_scaler']
        self.hyperparameters_config = model_config['hyperparameters']

        self.pipeline = self._get_pipeline()
        self.hyperparam_search_trials = None
        self.hyperparam_splits = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparam_search_trials: Optional[int] = 0,
        hyperparam_splits: Optional[int] = 3,
    ):
        """
        Fit the model to the data, possibly with hyperparameter tuning.

        Args:
            X: pd.DataFrame, the training data
            y: pd.Series, the target variable
            hyperparam_search_trials: Optional[int], number of trials for hyperparameter search
            hyperparam_splits: Optional[int], number of splits for cross-validation
        """
        self.hyperparam_search_trials = hyperparam_search_trials
        self.hyperparam_splits = hyperparam_splits

        if self.hyperparam_search_trials == 0:
            logger.info(
                'No hyperparam search trials provided, fitting the model with default hyperparameters'
            )
            self.pipeline.fit(X, y)

        else:
            logger.info(
                f"Let's find the best hyperparams for the model with {self.hyperparam_search_trials} trials"
            )
            best_hyperparams = self._find_best_hyperparams(X, y)
            logger.info(f'Best hyperparams: {best_hyperparams}')
            self.pipeline = self._get_pipeline(best_hyperparams)
            logger.info('Fitting the model with the best hyperparams')
            self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target variable.
        """
        return self.pipeline.predict(X)

    def _get_pipeline(self, model_hyperparams: Optional[dict] = None) -> Pipeline:
        """
        Get the pipeline for the model.
        """
        if model_hyperparams is None:
            if self.use_scaler:
                pipeline = Pipeline(
                    steps=[
                        ('preprocessor', StandardScaler()),
                        ('model', self.model_class()),
                    ]
                )
            else:
                pipeline = Pipeline(steps=[('model', self.model_class())])
        else:
            if self.use_scaler:
                pipeline = Pipeline(
                    steps=[
                        ('preprocessor', StandardScaler()),
                        ('model', self.model_class(**model_hyperparams)),
                    ]
                )
            else:
                pipeline = Pipeline(
                    steps=[('model', self.model_class(**model_hyperparams))]
                )
        return pipeline

    def _find_best_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """
        Finds the best hyperparameters for the model using Bayesian optimization.
        This method is now implemented in the base class and can be used by all subclasses.

        Args:
            X_train: pd.DataFrame, the training data
            y_train: pd.Series, the target variable

        Returns:
            dict, the best hyperparameters
        """
        # Create the objective function
        objective = self._create_objective_function(X_train, y_train)

        # Create a study object that minimizes the objective function
        study = optuna.create_study(direction='minimize')

        # Run the trials
        logger.info(f'Running {self.hyperparam_search_trials} trials')
        study.optimize(objective, n_trials=self.hyperparam_search_trials)

        # Return the best hyperparameters
        return study.best_trial.params

    def _create_objective_function(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Creates the objective function for Optuna optimization.
        This is a helper method that can be used by subclasses.

        Args:
            X_train: pd.DataFrame, the training data
            y_train: pd.Series, the target variable

        Returns:
            function: The objective function for Optuna
        """

        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for Optuna that returns the mean absolute error we
            want to minimize.

            Args:
                trial: optuna.Trial, the trial object

            Returns:
                float, the mean absolute error
            """
            # Get hyperparameters for this trial (loaded from JSON config)
            params = self._sample_hyperparameters(trial)

            # Split the training data into n_splits folds using a TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.hyperparam_splits or 1)
            mae_scores = []

            for train_index, val_index in tscv.split(X_train):
                # Split the data into training and validation sets
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_index],
                    X_train.iloc[val_index],
                )
                y_train_fold, y_val_fold = (
                    y_train.iloc[train_index],
                    y_train.iloc[val_index],
                )

                # Build a pipeline with preprocessing and model steps
                pipeline = self._get_pipeline(model_hyperparams=params)

                # Train the model on the training set
                pipeline.fit(X_train_fold, y_train_fold)

                # Evaluate the model on the validation set
                y_pred = pipeline.predict(X_val_fold)
                mae = mean_absolute_error(y_val_fold, y_pred)
                mae_scores.append(mae)

            # Return the average MAE across all folds
            return float(np.mean(mae_scores))

        return objective

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the given trial based on JSON configuration.

        Args:
            trial: optuna.Trial, the trial object

        Returns:
            dict, the sampled hyperparameters
        """
        params = {}

        # First pass: sample all non-conditional parameters
        for param_name, param_config in self.hyperparameters_config.items():
            param_type = param_config['type']

            if param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_type == 'float':
                if 'step' in param_config:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config['step']
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )

        # Second pass: handle conditional parameters
        for param_name, param_config in self.hyperparameters_config.items():
            param_type = param_config['type']

            if param_type == 'conditional_float':
                # Handle conditional parameters (like l1_ratio for SGDRegressor)
                condition = param_config['condition']
                if condition['param'] in params and params[condition['param']] == condition['value']:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )

        return params


def get_model_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_candidates: int,
) -> list[str]:
    """
    Uses lazypredict to fit N models with default hyperparameters for the given
    (X_train, y_train), and evaluate them with (X_test, y_test)

    It returns a list of model names, from best to worst.

    Args:
        X_train: pd.DataFrame, the training data
        y_train: pd.Series, the target variable
        X_test: pd.DataFrame, the test data
        y_test: pd.Series, the target variable
        n_candidates: int, the number of candidates to return

    Returns:
        list[str], the list of model names from best to worst
    """
    # unset the MLFLOW_TRACKING_URI
    # This is a temporary hack to avoid LazyPredict from
    # setting its own MLFLOW_TRACKING_URI. We want to use
    # the parent MLFlow run's tracking URI.
    # TODO: find a better way to do this.
    mlflow_tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    del os.environ['MLFLOW_TRACKING_URI']

    # fit N models with default hyperparameters
    reg = LazyRegressor(
        verbose=1,  # Show progress
        ignore_warnings=True,  # Ignore warnings to prevent crashes
        custom_metric=mean_absolute_error,
        predictions=True,  # Return predictions for efficiency
    )
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    # reset the index so that the model names are in the first column
    models.reset_index(inplace=True)

    # log table to mlflow experiment
    mlflow.log_table(models, 'model_candidates_with_default_hyperparameters.json')

    # set the MLFLOW_TRACKING_URI back to its original value
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

    # list of top n_candidates model names
    model_candidates = models['Model'].tolist()[:n_candidates]

    return model_candidates


Model = ModelWithHyperparameterTuning


def get_model_object(model_name: str) -> Model:
    """
    Get the model object based on the model name.
    Now uses the unified base class that loads configuration from JSON.

    Args:
        model_name: str, the name of the model

    Returns:
        ModelWithHyperparameterTuning: the model object

    Raises:
        ValueError: If the model is not found in the configuration
    """
    try:
        return ModelWithHyperparameterTuning(model_name)
    except ValueError as e:
        # Re-raise with more context
        raise NotImplementedError(str(e)) from e
