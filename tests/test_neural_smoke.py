import torch
from torch import nn

from kg_project.ner.neural import BertCRF, BiLSTMCRF


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        return self.embedding(input_ids)


def test_bilstm_crf_smoke() -> None:
    model = BiLSTMCRF(vocab_size=32, num_labels=5, embedding_dim=8, hidden_dim=16)
    input_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])

    loss = model.loss(input_ids, labels, mask)
    decoded = model.decode(input_ids, mask)

    assert loss.item() >= 0.0
    assert len(decoded[0]) == 3


def test_bert_crf_smoke_without_download() -> None:
    encoder = DummyEncoder(hidden_size=12)
    model = BertCRF(model_name="dummy", num_labels=4, encoder=encoder)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[True, True, True]])

    loss = model.loss(input_ids, attention_mask, labels, mask)
    decoded = model.decode(input_ids, attention_mask, mask)

    assert loss.item() >= 0.0
    assert len(decoded[0]) == 3
