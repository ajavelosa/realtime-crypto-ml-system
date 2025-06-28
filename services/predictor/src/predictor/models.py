"""
Machine Learning Models with Hyperparameter Tuning

This module provides a framework for creating machine learning models with automatic
hyperparameter tuning using Optuna for Bayesian optimization.

HOW TO CREATE HYPERPARAMETER-TUNED MODEL SUBCLASSES:

1. **Inherit from ModelWithHyperparameterTuning**:
   ```python
   class YourModelWithHyperparameterTuning(ModelWithHyperparameterTuning):
       def __init__(self):
           # For models that need scaling (linear models, SVM, neural networks, etc.)
           super().__init__(model_class=YourModelClass, use_scaler=True)

           # For tree-based models (Random Forest, XGBoost, etc.)
           super().__init__(model_class=YourModelClass, use_scaler=False)
   ```

2. **Implement _sample_hyperparameters method** (ONLY THIS METHOD IS REQUIRED):
   ```python
   def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
       return {
           'param1': trial.suggest_int('param1', 1, 100),
           'param2': trial.suggest_float('param2', 0.01, 1.0, log=True),
           'param3': trial.suggest_categorical('param3', ['option1', 'option2']),
       }
   ```

3. **Add to get_model_object function**:
   ```python
   elif model_name == 'YourModel':
       return YourModelWithHyperparameterTuning()
   ```

4. **Update Model type union**:
   ```python
   Model = Union['YourModelWithHyperparameterTuning', ...]
   ```

SUPPORTED MODELS:
    Only predefined models with hyperparameter tuning are supported. If you need a new model,
    create a new subclass following the template above.

    Available models:
    - LinearRegression: Linear regression with hyperparameter tuning
    - OrthogonalMatchingPursuit: Orthogonal matching pursuit with hyperparameter tuning
    - HuberRegressor: Huber regression with hyperparameter tuning
    - SGDRegressor: Stochastic gradient descent with hyperparameter tuning
    - RandomForestRegressor: Random forest with hyperparameter tuning (no scaling)
    - PassiveAggressiveRegressor: Passive aggressive regression with hyperparameter tuning

WHEN TO USE STANDARDSCALER:
    - ✅ USE SCALING for: Linear models, SVM, Neural Networks, Gradient Descent
    - ❌ DON'T USE SCALING for: Tree-based models (Random Forest, XGBoost, Decision Trees)

    Tree-based models are scale-invariant and scaling can sometimes hurt performance.

USAGE:
    # Create model instance
    model = YourModelWithHyperparameterTuning()

    # Fit with hyperparameter tuning (100 trials, 3 CV splits)
    model.fit(X_train, y_train, hyperparam_search_trials=100, hyperparam_splits=3)

    # Fit without hyperparameter tuning (uses default parameters)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

AVAILABLE MODELS:
    - LinearRegressionWithHyperparameterTuning (uses scaling)
    - HuberRegressorWithHyperparameterTuning (uses scaling)
    - OrthogonalMatchingPursuitWithHyperparameterTuning (uses scaling)
    - SGDRegressorWithHyperparameterTuning (uses scaling)
    - RandomForestWithHyperparameterTuning (no scaling - tree-based)
    - PassiveAggressiveRegressorWithHyperparameterTuning (uses scaling)
    - TemplateModelWithHyperparameterTuning (template for new models)
"""

import os
from typing import Optional, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    HuberRegressor,
    LarsCV,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    SGDRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    """

    def __init__(self, model_class: type, use_scaler: bool = True):
        self.model_class = model_class
        self.use_scaler = use_scaler
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
            # Get hyperparameters for this trial (to be implemented by subclasses)
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
        Sample hyperparameters for the given trial.
        This method should be overridden by subclasses to define specific hyperparameter spaces.

        Args:
            trial: optuna.Trial, the trial object

        Returns:
            dict, the sampled hyperparameters
        """
        raise NotImplementedError('Subclasses must implement _sample_hyperparameters')


class SGDRegressorWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Fits a SGDRegressor with hyperparameter tuning.
    """

    def __init__(self):
        super().__init__(model_class=SGDRegressor)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the SGDRegressor.
        """
        # Sample penalty first since it affects other parameters
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])

        # Base parameters
        params = {
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'learning_rate': trial.suggest_categorical(
                'learning_rate', ['constant', 'invscaling', 'optimal', 'adaptive']
            ),
            'eta0': trial.suggest_float('eta0', 0.001, 1.0, log=True),
            'penalty': penalty,
        }

        # Only add l1_ratio if penalty is elasticnet
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        return params


class HuberRegressorWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Fits a HuberRegressor with hyperparameter tuning.
    """

    def __init__(self):
        super().__init__(model_class=HuberRegressor)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the HuberRegressor.

        Args:
            trial: optuna.Trial, the trial object

        Returns:
            dict, the sampled hyperparameters
        """
        return {
            'epsilon': trial.suggest_float('epsilon', 1.0, 2.0),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        }


class OrthogonalMatchingPursuitWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Orthogonal Matching Pursuit with hyperparameter tuning.
    """

    def __init__(self):
        super().__init__(model_class=OrthogonalMatchingPursuit)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the Orthogonal Matching Pursuit model.

        Args:
            trial: optuna.Trial, the trial object

        Returns:
            dict, the sampled hyperparameters
        """
        return {
            'n_nonzero_coefs': trial.suggest_int('n_nonzero_coefs', 1, 10),
            'tol': trial.suggest_float('tol', 1e-4, 1e-1, step=1e-4),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }


class LarsCVWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    LarsCV with hyperparameter tuning.
    """

    def __init__(self):
        super().__init__(model_class=LarsCV)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the LarsCV model.
        """
        return {
            'max_n_alphas': trial.suggest_int('max_n_alphas', 10, 100),
            'eps': trial.suggest_float('eps', 1e-4, 1e-1, step=1e-4),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'n_jobs': trial.suggest_categorical('n_jobs', [-1, 1]),
            'precompute': trial.suggest_categorical('precompute', [True, False]),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'cv': trial.suggest_int('cv', 2, 10),
        }


# Template for creating new hyperparameter-tuned model subclasses
class TemplateModelWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Template class showing how to create a new hyperparameter-tuned model.
    Replace 'TemplateModel' with your actual model class.

    NOTE: Only _sample_hyperparameters method needs to be implemented!
    The _find_best_hyperparams method is now handled by the base class.
    """

    def __init__(self):
        # Choose the appropriate preprocessing strategy:

        # For models that NEED scaling (linear models, SVM, neural networks, etc.)
        # from sklearn.linear_model import LinearRegression
        # super().__init__(model_class=LinearRegression, use_scaler=True)

        # For models that DON'T need scaling (tree-based models)
        # from sklearn.ensemble import RandomForestRegressor
        # super().__init__(model_class=RandomForestRegressor, use_scaler=False)

        pass

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for your model.
        Define the hyperparameter space here using trial.suggest_* methods.
        THIS IS THE ONLY METHOD YOU NEED TO IMPLEMENT!

        Args:
            trial: optuna.Trial, the trial object

        Returns:
            dict, the sampled hyperparameters
        """
        return {
            # Example hyperparameters (replace with your model's actual parameters):
            # 'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            # 'max_depth': trial.suggest_int('max_depth', 1, 20),
            # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            # 'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        }


class LinearRegressionWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Linear Regression with hyperparameter tuning.
    Note: LinearRegression has very few hyperparameters, so this is mainly for consistency.
    """

    def __init__(self):
        super().__init__(model_class=LinearRegression)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the Linear Regression.
        LinearRegression has very few hyperparameters.
        """
        return {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'n_jobs': trial.suggest_categorical('n_jobs', [-1, 1]),
            'positive': trial.suggest_categorical('positive', [True, False]),
        }


# Practical example: Random Forest with hyperparameter tuning
class RandomForestWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Random Forest Regressor with hyperparameter tuning.
    Tree-based models like Random Forest don't need feature scaling.
    """

    def __init__(self):
        super().__init__(model_class=RandomForestRegressor, use_scaler=False)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the Random Forest.

        Args:
            trial: optuna.Trial, the trial object

        Returns:
            dict, the sampled hyperparameters
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]
            ),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }


class PassiveAggressiveRegressorWithHyperparameterTuning(ModelWithHyperparameterTuning):
    """
    Passive Aggressive Regressor with hyperparameter tuning.
    """

    def __init__(self):
        super().__init__(model_class=PassiveAggressiveRegressor)

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Sample hyperparameters for the Passive Aggressive Regressor.
        """
        return {
            'C': trial.suggest_float('C', 0.001, 1.0, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'epsilon': trial.suggest_float('epsilon', 0.0, 1.0),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'loss': trial.suggest_categorical(
                'loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']
            ),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }


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
        verbose=0,
        ignore_warnings=False,
        custom_metric=mean_absolute_error,
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


Model = Union[
    'LinearRegressionWithHyperparameterTuning',
    'OrthogonalMatchingPursuitWithHyperparameterTuning',
    'HuberRegressorWithHyperparameterTuning',
    'SGDRegressorWithHyperparameterTuning',
    'RandomForestWithHyperparameterTuning',
    'PassiveAggressiveRegressorWithHyperparameterTuning',
    'LarsCVWithHyperparameterTuning',
]


def get_model_object(model_name: str) -> Model:
    """
    Get the model object based on the model name.
    Only predefined models are supported.

    Args:
        model_name: str, the name of the model

    Returns:
        Model, the model object

    Raises:
        NotImplementedError: If the model is not among the available predefined models
    """
    # Predefined model mappings
    predefined_models = {
        'LinearRegression': LinearRegressionWithHyperparameterTuning,
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuitWithHyperparameterTuning,
        'HuberRegressor': HuberRegressorWithHyperparameterTuning,
        'SGDRegressor': SGDRegressorWithHyperparameterTuning,
        'RandomForestRegressor': RandomForestWithHyperparameterTuning,
        'PassiveAggressiveRegressor': PassiveAggressiveRegressorWithHyperparameterTuning,
        'LarsCV': LarsCVWithHyperparameterTuning,
    }

    # Try to get from predefined models
    if model_name in predefined_models:
        return predefined_models[model_name]()

    # If model not found, raise NotImplementedError
    available_models = list(predefined_models.keys())
    raise NotImplementedError(
        f'Model "{model_name}" is not implemented. Available models: {", ".join(available_models)}'
    )
