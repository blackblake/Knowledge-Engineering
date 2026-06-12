from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only when training deps are absent.
    torch = None
    Tensor = Any
    nn = None

try:
    from transformers import AutoModel
except ImportError:  # pragma: no cover - exercised only when training deps are absent.
    AutoModel = None


class MissingTrainingDependencyError(RuntimeError):
    pass


def ensure_training_dependencies() -> None:
    if torch is None or nn is None:
        raise MissingTrainingDependencyError(
            "Torch/Transformers are not installed. Install with `pip install -e \".[training]\"`."
        )


if torch is not None and nn is not None:

    class LinearChainCRF(nn.Module):
        def __init__(self, num_tags: int) -> None:
            super().__init__()
            self.num_tags = num_tags
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)

        def _compute_score(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
            batch_size, seq_len, _ = emissions.shape
            score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1)
            for step in range(1, seq_len):
                transition_score = self.transitions[tags[:, step - 1], tags[:, step]]
                emission_score = emissions[:, step, :].gather(1, tags[:, step : step + 1]).squeeze(1)
                score = score + (transition_score + emission_score) * mask[:, step]
            seq_ends = mask.long().sum(dim=1) - 1
            last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
            return score + self.end_transitions[last_tags]

        def _compute_log_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
            score = self.start_transitions + emissions[:, 0]
            for step in range(1, emissions.size(1)):
                next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, step].unsqueeze(1)
                next_score = torch.logsumexp(next_score, dim=1)
                score = torch.where(mask[:, step].unsqueeze(1), next_score, score)
            return torch.logsumexp(score + self.end_transitions, dim=1)

        def forward(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
            log_denominator = self._compute_log_partition(emissions, mask)
            log_numerator = self._compute_score(emissions, tags, mask)
            return torch.mean(log_denominator - log_numerator)

        def decode(self, emissions: Tensor, mask: Tensor) -> list[list[int]]:
            batch_size, seq_len, num_tags = emissions.shape
            score = self.start_transitions + emissions[:, 0]
            history: list[Tensor] = []
            for step in range(1, seq_len):
                next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
                best_score, best_path = torch.max(next_score, dim=1)
                candidate_score = best_score + emissions[:, step]
                score = torch.where(mask[:, step].unsqueeze(1), candidate_score, score)
                history.append(best_path)

            score = score + self.end_transitions
            best_last_score, best_last_tag = torch.max(score, dim=1)
            del best_last_score
            best_paths: list[list[int]] = []
            lengths = mask.long().sum(dim=1)
            for batch_idx in range(batch_size):
                length = lengths[batch_idx].item()
                best_tag = best_last_tag[batch_idx].item()
                path = [best_tag]
                for hist in reversed(history[: max(length - 1, 0)]):
                    best_tag = hist[batch_idx][best_tag].item()
                    path.append(best_tag)
                best_paths.append(list(reversed(path)))
            return best_paths


    class BiLSTMCRF(nn.Module):
        def __init__(self, vocab_size: int, num_labels: int, embedding_dim: int = 128, hidden_dim: int = 128) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(hidden_dim, num_labels)
            self.crf = LinearChainCRF(num_labels)

        def emissions(self, input_ids: Tensor) -> Tensor:
            embedded = self.embedding(input_ids)
            encoded, _ = self.encoder(embedded)
            return self.classifier(self.dropout(encoded))

        def loss(self, input_ids: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
            return self.crf(self.emissions(input_ids), tags, mask)

        def decode(self, input_ids: Tensor, mask: Tensor) -> list[list[int]]:
            return self.crf.decode(self.emissions(input_ids), mask)


    class BertCRF(nn.Module):
        def __init__(self, model_name: str, num_labels: int, encoder: nn.Module | None = None) -> None:
            super().__init__()
            if encoder is not None:
                self.encoder = encoder
            else:
                if AutoModel is None:
                    raise MissingTrainingDependencyError(
                        "Transformers are not installed. Install with `pip install -e \".[training]\"`."
                    )
                self.encoder = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
            self.crf = LinearChainCRF(num_labels)

        def emissions(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded
            return self.classifier(self.dropout(hidden))

        def loss(self, input_ids: Tensor, attention_mask: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
            return self.crf(self.emissions(input_ids, attention_mask), tags, mask)

        def decode(self, input_ids: Tensor, attention_mask: Tensor, mask: Tensor) -> list[list[int]]:
            return self.crf.decode(self.emissions(input_ids, attention_mask), mask)


else:  # pragma: no cover - this path is for documentation/runtime fallback only.

    @dataclass(slots=True)
    class LinearChainCRF:
        num_tags: int


    @dataclass(slots=True)
    class BiLSTMCRF:
        vocab_size: int
        num_labels: int
        embedding_dim: int = 128
        hidden_dim: int = 128


    @dataclass(slots=True)
    class BertCRF:
        model_name: str
        num_labels: int
