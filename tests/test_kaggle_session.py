"""Unit tests for the Kaggle session orchestrator's download/verify and
prior-results ingestion, using file:// URLs — no network."""

import hashlib
import json
import zipfile

import pytest

from scripts.kaggle_session import download_and_verify, ingest_prior_results


def _cfg_for(path, sha256):
    return {"files": {"Q8_0": {"gguf_url": path.as_uri(), "sha256": sha256}}}


def test_download_and_verify_ok(tmp_path):
    src = tmp_path / "src" / "tiny.gguf"
    src.parent.mkdir()
    src.write_bytes(b"fake gguf bytes")
    good = hashlib.sha256(b"fake gguf bytes").hexdigest()
    models_dir = tmp_path / "models"
    out = download_and_verify(_cfg_for(src, good), "Q8_0", models_dir)
    assert out == models_dir / "tiny.gguf"
    assert out.read_bytes() == b"fake gguf bytes"
    # second call: file exists, verify-only path, still OK
    assert download_and_verify(_cfg_for(src, good), "Q8_0", models_dir) == out


def test_hash_mismatch_aborts_and_keeps_file(tmp_path):
    src = tmp_path / "src" / "tiny.gguf"
    src.parent.mkdir()
    src.write_bytes(b"fake gguf bytes")
    models_dir = tmp_path / "models"
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        download_and_verify(_cfg_for(src, "0" * 64), "Q8_0", models_dir)
    # rule 5: no silent redownload/delete — the bad file is left in place
    assert (models_dir / "tiny.gguf").exists()


def test_unrecorded_hash_aborts(tmp_path):
    src = tmp_path / "tiny.gguf"
    src.write_bytes(b"x")
    with pytest.raises(SystemExit, match="rule 5"):
        download_and_verify(_cfg_for(src, "TODO"), "Q8_0", tmp_path / "m")


def test_ingest_prior_results_zip_and_dir(tmp_path):
    results = tmp_path / "results"
    # a prior zip with two records, one of which also exists locally already
    zpath = tmp_path / "results_smoke.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("aaa.json", json.dumps({"k": 1}))
        zf.writestr("bbb.json", json.dumps({"k": 2}))
    results.mkdir()
    (results / "aaa.json").write_text('{"k": "local"}')
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    (prior_dir / "ccc.json").write_text('{"k": 3}')

    ingest_prior_results([zpath, prior_dir], results)

    assert sorted(p.name for p in results.glob("*.json")) == [
        "aaa.json", "bbb.json", "ccc.json"]
    # existing files are never overwritten
    assert json.loads((results / "aaa.json").read_text()) == {"k": "local"}


def test_missing_prior_source_aborts(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        ingest_prior_results([tmp_path / "nope.zip"], tmp_path / "r")
