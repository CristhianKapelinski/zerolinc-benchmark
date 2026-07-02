"""Unit tests for the non-neural logic: labels, normalization, metrics, baselines.

These run offline (no network, no models).
"""

from zerolinc.baselines import keyword_baseline, majority_baseline
from zerolinc.data import normalize_text
from zerolinc.labels import CATEGORIES, CODES, PROMPT_CONFIGS
from zerolinc.metrics import evaluate, wilson_ci


def test_twelve_categories_and_codes():
    assert len(CATEGORIES) == 12
    assert CODES[0] == "CAT1" and CODES[-1] == "CAT12"
    assert len(set(CODES)) == 12


def test_prompt_configs_cover_all_categories():
    for name, cfg in PROMPT_CONFIGS.items():
        assert "{}" in cfg.template, name
        assert sorted(cfg.labels.values()) == sorted(CODES), name
        assert all(label.strip() for label in cfg.labels), name


def test_normalize_compresses_tags():
    raw = "De [EMAIL_ADDRESS_f6f7086365] host [IP_ADDRESS_907d29be4d]  em [DATE_TIME_1fe1abe111]"
    out = normalize_text(raw)
    assert out == "De <EMAIL> host <IP> em <DATE>"


def test_normalize_keeps_unknown_tag_safe():
    assert normalize_text("[FOO_BAR_abcdef1234]") == "<REDACTED>"


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_evaluate_counts_and_perfect_run():
    y = ["CAT1", "CAT2", "CAT2"]
    res = evaluate(y, y)
    assert res["accuracy"] == 1.0 and res["correct"] == 3
    res2 = evaluate(y, ["CAT2", "CAT2", "CAT2"])
    assert res2["correct"] == 2
    assert res2["per_class"]["CAT1"]["support"] == 1


def test_majority_baseline():
    y = ["CAT5", "CAT5", "CAT3"]
    assert majority_baseline(y, ["a", "b", "c"]) == ["CAT5", "CAT5", "CAT5"]


def test_keyword_baseline_matches_and_fallback():
    y = ["CAT3", "CAT5", "CAT5"]
    texts = [
        "massive ddos with syn flood from a botnet",  # clear CAT3
        "exploit for cve-2023-1234 remote execution",  # clear CAT5
        "nothing relevant here",  # no match -> majority (CAT5)
    ]
    assert keyword_baseline(y, texts) == ["CAT3", "CAT5", "CAT5"]


def test_parse_spec():
    from zerolinc.backends import parse_spec
    assert parse_spec("facebook/bart-large-mnli") == ("nli", "facebook/bart-large-mnli")
    assert parse_spec("nli:a/b") == ("nli", "a/b")
    assert parse_spec("gliclass:knowledgator/gliclass-x-base") == (
        "gliclass", "knowledgator/gliclass-x-base")
    assert parse_spec("embed:Qwen/Qwen3-Embedding-0.6B") == ("embed", "Qwen/Qwen3-Embedding-0.6B")


def test_subject_view_reorders_only():
    from zerolinc.data import subject_view
    text = "Pedido: <DATE> CERT.br: Assunto: maquina comprometida\ncorpo do relato"
    out = subject_view(text)
    assert out.startswith("Assunto: maquina comprometida.")
    assert "corpo do relato" in out and "Pedido:" in out
    assert subject_view("sem cabecalho") == "sem cabecalho"


def test_event_config_present():
    from zerolinc.labels import PROMPT_CONFIGS, CODES
    cfg = PROMPT_CONFIGS["en-event"]
    assert cfg.template == "{}"
    assert sorted(cfg.labels.values()) == sorted(CODES)
    assert all(lbl.endswith(".") for lbl in cfg.labels)


def test_deboiler_removes_corpus_templates():
    from zerolinc.data import Incident, apply_view
    boiler = "CERT.br works as a coordinating team for incidents"
    incs = [Incident(str(k), f"{boiler}\nconteudo unico {k}", "CAT5") for k in range(10)]
    out = apply_view(incs, "deboiler")
    assert all(boiler not in i.text for i in out)
    assert all(f"conteudo unico {k}" in out[k].text for k in range(10))
    # a text that would become empty falls back to the original
    incs2 = [Incident(str(k), boiler, "CAT5") for k in range(10)]
    assert all(i.text for i in apply_view(incs2, "deboiler"))


def test_calibration_removes_label_bias(tmp_path):
    import json
    from zerolinc.combine import evaluate_calibrated, evaluate_ensemble
    from zerolinc.labels import CODES
    # CAT9 has a +0.4 constant bias; true signal puts CAT5 on top for all items
    preds = []
    for k in range(10):
        scores = {c: 0.1 for c in CODES}
        scores["CAT5"] = 0.3   # real signal
        scores["CAT9"] = 0.5   # biased label wins raw argmax
        preds.append({"incident_id": str(k), "true": "CAT5", "pred": "CAT9",
                      "score": 0.5, "scores": scores})
    f = tmp_path / "run.json"
    f.write_text(json.dumps({"predictions": preds}))
    res = evaluate_calibrated(f)
    # constant bias is removed; with zero variance after centering CAT9 ties at 0,
    # CAT5 keeps positive margin only if variance exists; here all rows identical ->
    # calibrated scores all zero => argmax falls to first code CAT1: accuracy 0.
    # Add one contrast row to give variance instead:
    preds[0]["scores"]["CAT5"] = 0.05
    preds[0]["scores"]["CAT9"] = 0.9
    preds[0]["true"] = "CAT9"
    f.write_text(json.dumps({"predictions": preds}))
    res = evaluate_calibrated(f)
    assert res["accuracy"] >= 0.9  # 9 CAT5 rows + 1 CAT9 row all correct
    res2 = evaluate_ensemble([f, f])
    assert res2["accuracy"] == res["accuracy"]


def test_knn_vote_and_report():
    import numpy as np
    from zerolinc.data import Incident
    from zerolinc.knn import knn_report, _vote
    # 3 template clusters: CAT5-like, CAT3-like, CAT12-like; 20 items each
    rng = np.random.default_rng(0)
    labels = ["CAT5"] * 20 + ["CAT3"] * 20 + ["CAT12"] * 20
    centers = {"CAT5": [1, 0, 0], "CAT3": [0, 1, 0], "CAT12": [0, 0, 1]}
    emb = np.array([centers[lab] for lab in labels], dtype=float)
    emb += rng.normal(0, 0.05, emb.shape)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    incs = [Incident(str(i), f"texto {i}", lab) for i, lab in enumerate(labels)]
    r = knn_report(incs, {"full": emb}, seed=42, ks=(1, 3))
    assert r["test"]["accuracy"] > 0.95   # clustered templates are trivial for kNN
    assert r["selected_view"] == "full"
    # deterministic vote tie-break
    sims_row = np.array([0.9, 0.9])
    assert _vote(sims_row, ["CAT3", "CAT12"], [0, 1], 2) in ("CAT3", "CAT12")
