import os
import re
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the LightGBM Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "LightGBM Regressor"

    def __init__(
        self,
        boosting_type: Optional[str] = "gbdt",
        n_estimators: Optional[int] = 250,
        num_leaves: Optional[int] = 31,
        learning_rate: Optional[float] = 1e-1,
        **kwargs,
    ):
        """Construct a new LightGBM Regressor.

        Args:
            boosting_type (str, optional): The number of trees in the forest.
                Defaults to "gbdt".
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 100.
            num_leaves (int, optional): The minimum number of samples required
                to split an internal node.
                Defaults to 2.
            learning_rate (int, optional): The minimum number of samples required
                to be at a leaf node.
                Defaults to 1.
        """
        self.boosting_type = str(boosting_type)
        self.n_estimators = int(n_estimators)
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> LGBMRegressor:
        """Build a new regressor."""
        model = LGBMRegressor(
            objective="regression",
            num_class=1,
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            num_iterations=500,
            random_state=42
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        # lightgbm throws an error if column names contain special characters
        updated_train_inputs = \
            train_inputs.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        self.model.fit(updated_train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The targets of the test data.
        Returns:
            float: The r-squared score of the regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"boosting_type: {self.boosting_type}, "
            f"n_estimators: {self.n_estimators}, "
            f"num_leaves: {self.num_leaves}, "
            f"learning_rate: {self.learning_rate})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
