from inspect import getfullargspec, signature
from itertools import product, chain

import pandas as pd
from joblib import delayed, wrap_non_picklable_objects, Parallel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import (
    make_scorer,
    get_scorer,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from functools import reduce
from operator import or_

from util import DataLoader
from util import ResultHandler
from skpsl import ProbabilisticScoringList
from skpsl.preprocessing.binarizer import MinEntropyBinarizer
from skpsl.metrics import expected_entropy_loss, weighted_loss, soft_ranking_loss

RESULTFOLDER = "experiments/results"
DATAFOLDER = "experiments/data"


def estimator_factory(param):
    clf, *params = param
    params = [
        (
            (name, dict((v,)))
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str)
            else (name, v)
        )
        for name, v in params
    ]
    match clf:
        case "psl_prebin":
            kwargs = dict(params)
            pre = MinEntropyBinarizer(method=kwargs["method"])
            kwargs.pop("method")
            psl = ProbabilisticScoringList(**kwargs)
            return make_pipeline(
                SimpleImputer(missing_values=-1, strategy="most_frequent"), pre, psl
            )
        case "psl":
            psl = make_pipeline(
                SimpleImputer(missing_values=-1, strategy="most_frequent"),
                FunctionTransformer(),
                ProbabilisticScoringList(**dict(params)),
            )
            return psl
        case _:
            raise ValueError(f"classifier {clf} not defined")


def conservative_weighted_loss(y_true, y_prob, m=10, *, sample_weight=None):
    lb, _, ub = y_prob.T
    tn, fp, fn, tp = confusion_matrix(
        y_true, 1 - ub < m * ub, sample_weight=sample_weight, normalize="all"
    ).ravel()
    return fp + m * fn


# 1/m - ub/m < ub


def worker_facory():
    rh = ResultHandler(RESULTFOLDER)

    # all scorers that can be run on training and test independendly
    scoring = dict(
        acc=get_scorer("accuracy"),
        bacc=get_scorer("balanced_accuracy"),
        roc=get_scorer("roc_auc"),
        brier=get_scorer("neg_brier_score"),
        recall=get_scorer("recall"),
        recall_at_wloss=make_scorer(
            lambda true, pred: recall_score(true, 1 - pred < 10 * pred),
            response_method="predict_proba",
        ),
        prec=get_scorer("precision"),
        spec_at_wloss=make_scorer(
            lambda true, pred: recall_score(1 - true, 1 - pred >= 10 * pred),
            response_method="predict_proba",
        ),
        f1=get_scorer("f1"),
        ent=make_scorer(
            lambda _, pred: expected_entropy_loss(pred), response_method="predict_proba"
        ),
        wloss=make_scorer(weighted_loss, response_method="predict_proba"),
    )

    # this function can be used to calculate scoring functions, where the out-of-sample calculation needs to take decision thresholds
    # for the in-sample calculation into account
    def sample_aware_scorer(estimator, X_train, X_test, y_train, y_test):
        if "ci" in getfullargspec(estimator.predict_proba).args:
            dicts = [
                {
                    f"train_conservative_wloss{ci}": conservative_weighted_loss(
                        y_train, estimator.predict_proba(X_train, ci=ci / 100)
                    ),
                    f"test_conservative_wloss{ci}": conservative_weighted_loss(
                        y_test, estimator.predict_proba(X_test, ci=ci / 100)
                    ),
                }
                for ci in [5, 10, 20, 50, 80, 90, 95, 99]
            ]
            return reduce(or_, dicts, dict())
        return dict()

    @delayed
    @wrap_non_picklable_objects
    def worker(fold, dataset, params):
        key = (fold, dataset, params)
        rh.register_run(key)

        X, y = DataLoader(DATAFOLDER).load(dataset)

        clf = estimator_factory(params)
        results = cross_validate(
            clf,
            X,
            y,
            cv=ShuffleSplit(1, test_size=0.33, random_state=fold),
            n_jobs=1,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            return_indices=True,
        )

        (est,) = results["estimator"]
        indices = results["indices"]
        del results["estimator"]
        del results["indices"]

        (train,) = indices["train"]
        (test,) = indices["test"]
        y_train, y_test = y[train], y[test]
        X_train = est[:-1].transform(X[train])
        X_test = est[:-1].transform(X[test])
        psl = est[-1]

        results = pd.DataFrame(results).to_dict("records")
        results[0] |= sample_aware_scorer(psl, X_train, X_test, y_train, y_test)
        for k, stage in enumerate(psl):
            cur_results = (
                {
                    f"train_{name}": scorer(stage, X_train, y_train)
                    for name, scorer in scoring.items()
                }
                | {
                    f"test_{name}": scorer(stage, X_test, y_test)
                    for name, scorer in scoring.items()
                }
                | sample_aware_scorer(stage, X_train, X_test, y_train, y_test)
                | dict(stage=k)
            )
            results.append(cur_results)

            for penalty, name in [["l2", "logreg"], [None, "logreg_unregularized"]]:
                if k == 0:
                    # add performance of stage clf in to mimic performance of logreg at stage 0 (majority classifier)
                    cur_results = (
                        {
                            f"train_{name}": scorer(stage, X_train, y_train)
                            for name, scorer in scoring.items()
                        }
                        | {
                            f"test_{name}": scorer(stage, X_test, y_test)
                            for name, scorer in scoring.items()
                        }
                        | sample_aware_scorer(stage, X_train, X_test, y_train, y_test)
                        | dict(stage=k)
                        | dict(clf_variant=name)
                    )
                    results.append(cur_results)
                else:
                    X_train_ = X_train[:, stage.features]
                    X_test_ = X_test[:, stage.features]
                    logreg = make_pipeline(
                        SimpleImputer(missing_values=-1, strategy="most_frequent"),
                        LogisticRegression(max_iter=10000, penalty=penalty),
                    ).fit(X_train_, y_train)
                    cur_results = (
                        {
                            f"train_{name}": scorer(logreg, X_train_, y_train)
                            for name, scorer in scoring.items()
                        }
                        | {
                            f"test_{name}": scorer(logreg, X_test_, y_test)
                            for name, scorer in scoring.items()
                        }
                        | sample_aware_scorer(
                            logreg, X_train_, X_test_, y_train, y_test
                        )
                        | dict(stage=k)
                        | dict(clf_variant=name)
                    )
                    results.append(cur_results)
        rh.write_results(key, results)

    return worker


def dict_product(prefix, d):
    if not isinstance(prefix, list | tuple):
        prefix = [prefix]
    return [tuple(prefix + list(dict(zip(d, t)).items())) for t in product(*d.values())]


if __name__ == "__main__":
    datasets = ["thorax", 41945, 42900]
    splits = 100

    rh = ResultHandler(RESULTFOLDER)
    rh.clean()

    base = dict(
        score_set=[(-3, -2, -1, 1, 2, 3)],
        lookahead=[1],
        method=["bisect"],
        stage_clf_params=[("calibration_method", "isotonic")],
    )

    # create searchspace
    clf_params = chain(
        dict_product(prefix="psl_prebin", d=base | dict(method=["bisect", "brute"])),
        dict_product(prefix="psl", d=base | dict(method=["bisect", "brute"])),
        dict_product(
            prefix="psl",
            d=base
            | dict(
                stage_clf_params=[
                    ("calibration_method", "isotonic"),
                    ("calibration_method", "sigmoid"),
                    ("calibration_method", "beta"),
                    ("calibration_method", "beta_reg"),
                ]
            ),
        ),
        dict_product(
            prefix="psl_prebin",
            d=base
            | dict(
                stage_clf_params=[
                    ("calibration_method", "isotonic"),
                    ("calibration_method", "sigmoid"),
                    ("calibration_method", "beta"),
                    ("calibration_method", "beta_reg"),
                ]
            ),
        ),
        dict_product(
            prefix="psl_prebin",
            d=base
            | dict(
                score_set=[
                    (-3, -2, -1),
                    (-2, -1),
                    (1,),
                    (1, 2),
                    (1, 2, 3),
                    (-3, -2, -1, 1, 2, 3),
                ]
            ),
        ),
        dict_product(
            prefix="psl",
            d=base
            | dict(
                score_set=[
                    (-3, -2, -1, 1, 2, 3),
                    (-2, -1, 1, 2),
                    (-1, 1),
                ]
            ),
        ),
        dict_product(prefix="psl", d=base | dict(stage_loss=[soft_ranking_loss])),
    )

    grid = list(
        filter(
            rh.is_unprocessed,
            dict.fromkeys(product(range(splits), datasets, clf_params)),
        )
    )

    worker = worker_facory()
    list(
        tqdm(
            Parallel(n_jobs=10, return_as="generator_unordered")(
                worker(fold, dataset, params) for fold, dataset, params in grid
            ),
            total=len(grid),
        )
    )
