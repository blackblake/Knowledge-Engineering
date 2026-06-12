from __future__ import annotations

from dataclasses import dataclass

import sklearn_crfsuite
from seqeval.metrics import f1_score, precision_score, recall_score

from kg_project.data_types import GazetteerEntry, NerExample
from kg_project.ner.features import GazetteerMatcher, compile_gazetteer_entries, sent2features


@dataclass(slots=True)
class CRFReport:
    precision: float
    recall: float
    f1: float
    num_examples: int


class CRFTagger:
    def __init__(
        self,
        *,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        use_gazetteer: bool = False,
    ) -> None:
        self.use_gazetteer = use_gazetteer
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def _build_features(
        self,
        examples: list[NerExample],
        gazetteer_entries: list[GazetteerEntry] | None = None,
        gazetteer_matcher: GazetteerMatcher | None = None,
    ) -> list[list[dict[str, object]]]:
        matcher = gazetteer_matcher
        if self.use_gazetteer and matcher is None:
            matcher = compile_gazetteer_entries(gazetteer_entries)
        return [
            sent2features(
                example.tokens,
                gazetteer_entries=gazetteer_entries,
                matcher=matcher,
                use_gazetteer=self.use_gazetteer,
            )
            for example in examples
        ]

    def fit(
        self,
        examples: list[NerExample],
        gazetteer_entries: list[GazetteerEntry] | None = None,
        gazetteer_matcher: GazetteerMatcher | None = None,
    ) -> None:
        x_train = [
            features
            for features in self._build_features(
                examples,
                gazetteer_entries=gazetteer_entries,
                gazetteer_matcher=gazetteer_matcher,
            )
        ]
        y_train = [example.labels for example in examples]
        self.model.fit(x_train, y_train)

    def predict(
        self,
        examples: list[NerExample],
        gazetteer_entries: list[GazetteerEntry] | None = None,
        gazetteer_matcher: GazetteerMatcher | None = None,
    ) -> list[list[str]]:
        x_eval = self._build_features(
            examples,
            gazetteer_entries=gazetteer_entries,
            gazetteer_matcher=gazetteer_matcher,
        )
        return self.model.predict(x_eval)

    @staticmethod
    def score_predictions(y_true: list[list[str]], y_pred: list[list[str]]) -> CRFReport:
        return CRFReport(
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            num_examples=len(y_true),
        )

    def evaluate(
        self,
        examples: list[NerExample],
        gazetteer_entries: list[GazetteerEntry] | None = None,
        gazetteer_matcher: GazetteerMatcher | None = None,
    ) -> CRFReport:
        y_true = [example.labels for example in examples]
        y_pred = self.predict(
            examples,
            gazetteer_entries=gazetteer_entries,
            gazetteer_matcher=gazetteer_matcher,
        )
        return self.score_predictions(y_true, y_pred)
