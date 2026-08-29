"""Uplift meta-learners, implemented directly on scikit-learn.

Deliberately not a wrapper around `causalml`: the whole point of the project is
being able to explain what each estimator does, and these are twenty lines each.

    SLearner              one model, treatment as a feature
    TLearner              two models, subtract
    XLearner              impute the effect, cross-fit, blend by propensity
    TransformedOutcome    Athey-Imbens modified outcome; a single regression
                          whose conditional mean IS the treatment effect
                          (the propensity-corrected generalisation of the
                          Lai / Kane class-transformation trick)

All four expose the same surface: .fit(X, treatment, y) then .predict(X) -> tau.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# LightGBM is the production choice, but it is an optional accelerator here:
# sklearn's histogram gradient boosting is the same algorithm family and ships
# with the environment, so the project never fails to run over a build toolchain.
try:  # pragma: no cover
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

RANDOM_STATE = 7


def make_classifier():
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=260, learning_rate=0.05, num_leaves=31,
            min_child_samples=60, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=260, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def make_regressor():
    if HAS_LGBM:
        return LGBMRegressor(
            n_estimators=260, learning_rate=0.05, num_leaves=31,
            min_child_samples=60, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=260, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class UpliftLearner:
    """Common interface."""

    name = "base"

    def fit(self, X, t, y):  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_control(self, X) -> np.ndarray:
        """P(convert | untreated). Needed to separate sure things from lost causes."""
        raise NotImplementedError


class SLearner(UpliftLearner):
    """Single model with treatment appended as a feature.

    Cheap, but a tree can simply decline to split on the treatment column, which
    shrinks the estimated effect toward zero. Included because seeing that
    failure mode is more instructive than reading about it.
    """

    name = "S-Learner"

    def fit(self, X, t, y):
        X = np.asarray(X, dtype=float)
        Xt = np.column_stack([X, np.asarray(t, dtype=float)])
        self.model_ = make_classifier().fit(Xt, y)
        return self

    def _both(self, X):
        X = np.asarray(X, dtype=float)
        x1 = np.column_stack([X, np.ones(len(X))])
        x0 = np.column_stack([X, np.zeros(len(X))])
        return _proba(self.model_, x1), _proba(self.model_, x0)

    def predict(self, X):
        p1, p0 = self._both(X)
        return p1 - p0

    def predict_control(self, X):
        return self._both(X)[1]


class TLearner(UpliftLearner):
    """Two independent models, differenced. Unbiased but high variance -- each
    model only sees its own arm, so errors do not cancel."""

    name = "T-Learner"

    def fit(self, X, t, y):
        X, t, y = np.asarray(X, float), np.asarray(t), np.asarray(y)
        self.m1_ = make_classifier().fit(X[t == 1], y[t == 1])
        self.m0_ = make_classifier().fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return _proba(self.m1_, X) - _proba(self.m0_, X)

    def predict_control(self, X):
        return _proba(self.m0_, np.asarray(X, float))


class XLearner(UpliftLearner):
    """Kuenzel et al. Imputes the individual effect in each arm using the other
    arm's model, fits a regressor to those imputed effects, then blends the two
    by propensity. This is what makes it hold up when the control group is much
    smaller than the treated group -- exactly our 25/75 split."""

    name = "X-Learner"

    def fit(self, X, t, y):
        X, t, y = np.asarray(X, float), np.asarray(t), np.asarray(y, float)
        X1, y1 = X[t == 1], y[t == 1]
        X0, y0 = X[t == 0], y[t == 0]

        self.m1_ = make_classifier().fit(X1, y1)
        self.m0_ = make_classifier().fit(X0, y0)

        # impute each unit's effect using the opposite arm's outcome model
        d1 = y1 - _proba(self.m0_, X1)
        d0 = _proba(self.m1_, X0) - y0

        self.tau1_ = make_regressor().fit(X1, d1)
        self.tau0_ = make_regressor().fit(X0, d0)

        self.e_ = make_classifier().fit(X, t)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        g = np.clip(_proba(self.e_, X), 0.05, 0.95)
        return g * self.tau0_.predict(X) + (1.0 - g) * self.tau1_.predict(X)

    def predict_control(self, X):
        return _proba(self.m0_, np.asarray(X, float))


class TransformedOutcome(UpliftLearner):
    """Athey & Imbens modified outcome.

        Y* = Y * (T - e(X)) / (e(X)(1 - e(X)))      =>      E[Y* | X] = tau(X)

    One regression, and its conditional mean is the treatment effect directly.
    Very fast and surprisingly competitive; the variance is the price, since the
    transform divides by a propensity that can get small.
    """

    name = "Transformed Outcome"

    def fit(self, X, t, y):
        X, t, y = np.asarray(X, float), np.asarray(t, float), np.asarray(y, float)
        self.e_ = make_classifier().fit(X, t)
        e = np.clip(_proba(self.e_, X), 0.08, 0.92)
        y_star = y * (t - e) / (e * (1.0 - e))
        self.tau_ = make_regressor().fit(X, y_star)
        self.m0_ = make_classifier().fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.tau_.predict(np.asarray(X, float))

    def predict_control(self, X):
        return _proba(self.m0_, np.asarray(X, float))


class PropensityBaseline(UpliftLearner):
    """NOT an uplift model. Predicts P(convert | contacted) on the treated arm
    only -- the model most teams actually ship. Kept so the evaluation can show
    that ranking by it targets sure things and produces near-zero incremental
    lift. This is the comparison the whole project exists to make."""

    name = "Propensity (baseline)"

    def fit(self, X, t, y):
        X, t, y = np.asarray(X, float), np.asarray(t), np.asarray(y)
        self.model_ = make_classifier().fit(X[t == 1], y[t == 1])
        self.m0_ = make_classifier().fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return _proba(self.model_, np.asarray(X, float))

    def predict_control(self, X):
        return _proba(self.m0_, np.asarray(X, float))


LEARNERS = {
    "propensity": PropensityBaseline,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "transformed": TransformedOutcome,
}
