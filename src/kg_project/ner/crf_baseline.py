from __future__ import annotations

from dataclasses import dataclass

import sklearn_crfsuite
from seqeval.metrics import f1_score, precision_score, recall_score

from kg_project.data_types import NerExample
from kg_project.ner.features import sent2features


@dataclass(slots=True)
class CRFReport:
    precision: float
    recall: float
    f1: float
    num_examples: int


class CRFTagger:
    def __init__(self) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )

    def fit(self, examples: list[NerExample]) -> None:
        x_train = [sent2features(example.tokens) for example in examples]
        y_train = [example.labels for example in examples]
        self.model.fit(x_train, y_train)

    def predict(self, examples: list[NerExample]) -> list[list[str]]:
        x_eval = [sent2features(example.tokens) for example in examples]
        return self.model.predict(x_eval)

    def evaluate(self, examples: list[NerExample]) -> CRFReport:
        y_true = [example.labels for example in examples]
        y_pred = self.predict(examples)
        return CRFReport(
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            num_examples=len(examples),
        )
