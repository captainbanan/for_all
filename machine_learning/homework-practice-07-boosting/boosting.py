from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from IPython.display import clear_output

from warnings import filterwarnings

import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

sns.set(style='darkgrid')
filterwarnings('ignore')

sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 0])


class Boosting:

    def __init__(
            self,
            base_model_class,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        self.validation_loss = np.full(self.n_estimators, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        model = self.base_model_class(**self.base_model_params).fit(x, -self.loss_derivative(y, predictions))
        self.gammas.append(self.find_optimal_gamma(y, predictions, model.predict(x)))
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for i in range(self.n_estimators):
            idx = np.random.choice(np.arange(len(x_train)), int(self.subsample * y_train.shape[0]), replace=True)
            self.fit_new_base_model(x_train[idx], y_train[idx], train_predictions[idx])

            train_predictions = self.predict_proba(x_train)[:, 0]
            self.validation_loss[i] = self.loss_fn(y_valid, self.predict_proba(x_valid)[:, 0])
            if i > 0:
                if self.early_stopping_rounds is not None and self.validation_loss[i] >= self.validation_loss[i - 1]:
                    break

        if self.plot:
            clear_output()
            plt.plot(self.validation_loss)
            plt.xlabel('base_models')
            plt.ylabel('validation_loss')
            plt.title('validation loss history')

    def predict_proba(self, x):
        predict = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predict += self.learning_rate * gamma * model.predict(x)

        return np.array((self.sigmoid(predict), 1 - self.sigmoid(predict))).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
