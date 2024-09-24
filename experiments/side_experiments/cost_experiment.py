import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skpsl import ProbabilisticScoringList
from sklearn.metrics import confusion_matrix
import networkx as nx
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from experiments.util import DataLoader

# fit psl and get a instance of the calibrator


def conservative_weighted_loss(y_true, y_prob, m=10, *, sample_weight=None):
    if y_prob.shape[1] == 3:
        lb, _, ub = y_prob.T
    else:
        ub = y_prob[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        1 - ub < m * ub,
        sample_weight=sample_weight,
        normalize="all",
        labels=[False, True],
    ).ravel()
    return fp + m * fn


def create_dag(psl):
    dag = nx.DiGraph()
    for stage, (first, second) in enumerate(zip(psl, psl[1:])):
        tscores = first.class_counts_per_score.keys()
        score = second.scores[-1]
        for tscore in tscores:
            first_neg = first.class_counts_per_score[tscore][0]
            first_pos = first.class_counts_per_score[tscore][1]
            first_proba = first.calibrator.transform(np.array([tscore])).item()
            second_0_neg = second.class_counts_per_score[tscore][0]
            second_0_pos = second.class_counts_per_score[tscore][1]
            second_0_proba = second.calibrator.transform(np.array([tscore])).item()
            second_1_neg = second.class_counts_per_score[tscore + score][0]
            second_1_pos = second.class_counts_per_score[tscore + score][1]
            second_1_proba = second.calibrator.transform(
                np.array([tscore + score])
            ).item()

            dag.add_node(
                (stage, tscore),
                num_neg=first_neg,
                num_pos=first_pos,
                proba=first_proba,
            )
            dag.add_node(
                (stage + 1, tscore),
                num_neg=second_0_neg,
                num_pos=second_0_pos,
                proba=second_0_proba,
            )
            dag.add_node(
                (stage + 1, tscore + score),
                num_neg=second_1_neg,
                num_pos=second_1_pos,
                proba=second_1_proba,
            )
            # case 1: Feature ist not active
            dag.add_edge((stage, tscore), (stage + 1, tscore), score=0)
            # case 1: Feature ist active
            dag.add_edge((stage, tscore), (stage + 1, tscore + score), score=score)
    return dag


def expected_loss(y_prob, m=10):
    ub = y_prob[:, 1].item()
    y_decision = 1 - ub < m * ub
    loss = (1 - ub) * 1 if y_decision else ub * 10
    return loss


def calculate_evois(dag, loss=expected_loss, weight_by_bucket_size=False):
    evois = {}
    for node in dag.nodes:

        proba = dag.nodes[node]["proba"]
        current_loss = loss(np.array([[1 - proba, proba]]))

        children = dag.successors(node)
        successors = nx.single_source_shortest_path_length(dag, node)
        largest_dist = max(successors.values())
        node_evois = {}
        for dist in range(1, largest_dist + 1):
            succs_at_dist = [k for k, v in successors.items() if int(v) == dist]
            succ_losses, weights = [], []
            for succ in succs_at_dist:
                succ_losses.append(
                    loss(
                        np.array(
                            [
                                [
                                    1 - dag.nodes[succ]["proba"],
                                    dag.nodes[succ]["proba"],
                                ]
                            ]
                        )
                    )
                )
                if weight_by_bucket_size:
                    weights.append(
                        dag.nodes[succ]["num_pos"] + dag.nodes[succ]["num_neg"]
                    )
                else:
                    weights.append(1)

            if succ_losses:
                succ_losses, weights = np.array(succ_losses), np.array(weights)
                expected_loss_reduction = np.average(
                    np.full_like(succ_losses, current_loss) - succ_losses,
                    weights=weights,
                )
                node_evois[dist] = expected_loss_reduction / dist
        evois[node] = node_evois
    evois = defaultdict(dict, evois)
    return evois


def worker(
    seed,
    per_instance_budget=3,
    ci=0.5,
    voi=0.05,
    score_set={1, 2, 3},
):
    X, y = DataLoader("../data").load("thorax")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=seed
    )
    pipeline = make_pipeline(
        SimpleImputer(missing_values=-1, strategy="most_frequent"),
        ProbabilisticScoringList(score_set=score_set).fit(X, y),
    )

    pipeline.fit(X_train, y_train)
    imputer = pipeline[0]
    psl = pipeline[1]

    X_test = imputer.transform(X_test)

    overall_budget = per_instance_budget * len(y_test)
    # Predictions when always evaluation the same amount of features per instance
    y_prob_non_adaptive = pipeline.predict_proba(X_test, k=per_instance_budget, ci=ci)

    # Instantiate dag for heuristic
    dag = create_dag(psl=psl)
    evois = calculate_evois(dag)
    y_prob_adaptive = []
    remaining_budget = overall_budget
    imputer = pipeline[0]
    psl = pipeline[1]

    X_test = imputer.transform(X_test)

    for x_test in X_test:
        x_test = x_test.reshape(1, -1)
        stage = 0
        current_score = int(
            psl[stage]
            ._compute_total_scores(
                x_test,
                psl[stage].features,
                psl[stage].scores_,
                psl[stage].feature_thresholds,
            )
            .item()
        )
        current_proba = psl.predict_proba(x_test, k=stage, ci=ci)[0]

        while (
            any(evoi >= voi for evoi in evois[(stage, current_score)].values())
            and remaining_budget - stage > 0
        ):
            stage += 1
            current_score = (
                psl[stage]
                ._compute_total_scores(
                    x_test,
                    psl[stage].features,
                    psl[stage].scores_,
                    psl[stage].feature_thresholds,
                )
                .item()
            )
            current_proba = psl.predict_proba(x_test, k=stage, ci=ci)[0]

        remaining_budget -= stage

        y_prob_adaptive.append(current_proba)
    y_prob_adaptive = np.stack(y_prob_adaptive)

    cwloss_adap = conservative_weighted_loss(y_test, y_prob_adaptive)
    cwloss_non_adap = conservative_weighted_loss(y_test, y_prob_non_adaptive)

    if y_prob_non_adaptive.shape[1] == 3:
        lb, _, ub = y_prob_non_adaptive.T
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        1 - ub < m * ub,
        sample_weight=sample_weight,
        normalize="all",
        labels=[False, True],
    ).ravel()

    # bacc_adap = balanced_accuracy_score(y_test, y_prob_adaptive.argmax(axis=1))
    # bacc_non_adap = balanced_accuracy_score(y_test, y_prob_non_adaptive.argmax(axis=1))

    # acc_adap = accuracy_score(y_test, y_prob_adaptive.argmax(axis=1))
    # acc_non_adap = accuracy_score(y_test, y_prob_non_adaptive.argmax(axis=1))
    return [
        [
            "adaptive",
            seed,
            per_instance_budget,
            overall_budget,
            score_set,
            ci,
            voi,
            cwloss_adap,
            remaining_budget,
            # bacc_adap,
            # acc_adap,
        ],
        [
            "baseline",
            seed,
            per_instance_budget,
            overall_budget,
            score_set,
            ci,
            voi,
            cwloss_non_adap,
            0,
            # bacc_non_adap,
            # acc_non_adap,
        ],
    ]


if __name__ == "__main__":
    data = []
    for seed in tqdm(range(0, 100)):
        data += worker(seed)
    df = pd.DataFrame(
        data,
        columns=[
            "method",
            "seed",
            "per_instance_budget",
            "overall_budget",
            "score_set",
            "ci",
            "voi",
            "cwloss",
            "remaining_budget",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            # "bacc_non_adap",
            # "acc_non_adap",
        ],
    )
    df.to_csv("../results/cost_results.csv")
