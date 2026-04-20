from __future__ import annotations


def token2features(tokens: list[str], index: int) -> dict[str, object]:
    token = tokens[index]
    features: dict[str, object] = {
        "bias": 1.0,
        "token.lower": token.lower(),
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.prefix1": token[:1],
        "token.prefix2": token[:2],
        "token.suffix1": token[-1:],
        "token.suffix2": token[-2:],
        "token.length": len(token),
    }
    if index == 0:
        features["BOS"] = True
    else:
        prev = tokens[index - 1]
        features["-1:token.lower"] = prev.lower()
        features["-1:token.istitle"] = prev.istitle()
    if index == len(tokens) - 1:
        features["EOS"] = True
    else:
        nxt = tokens[index + 1]
        features["+1:token.lower"] = nxt.lower()
        features["+1:token.istitle"] = nxt.istitle()
    return features


def sent2features(tokens: list[str]) -> list[dict[str, object]]:
    return [token2features(tokens, idx) for idx in range(len(tokens))]
