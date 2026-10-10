from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_residual_cache_server.py"


def test_cache_requires_explicit_sigma_and_declares_quality_change():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "KINDRED_QWEN_CACHE_SKIP_SIGMAS must contain at least one sigma" in source
    assert '"quality_changing": True' in source
    assert "abs(sigma - target) <= SIGMA_TOLERANCE" in source
    assert "KINDRED_QWEN_CACHE_RUN_MODES must contain only stock or cache" in source


def test_cache_is_scoped_by_stream_shapes_and_bypasses_only_selected_calls():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "branch={state['branch']}" in source
    assert "img={img.shape[1]};txt={txt.shape[1]}" in source
    assert 'selected(state["sigma"])' in source
    assert 'state["key"] in state["cached"]' in source
    assert 'state["mode"] == "cache"' in source
    assert 'return txt + cached["txt"], img + cached["img"]' in source
    assert "return txt, img" in source


def test_cache_resets_between_sigma_sequences_and_rejects_unsupported_workflows():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'state["cached"].clear()' in source
    assert 'state["sigma"] > state["previous_sigma"] + SIGMA_TOLERANCE' in source
    assert "supports only batch-1 text-to-image without ControlNet" in source
    assert "Qwen residual cache run-mode sequence exhausted" in source


def test_experimental_cache_bust_node_returns_latent_unchanged():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'nodes.NODE_CLASS_MAPPINGS["KindredCacheBustLatent"]' in source
    assert "return (samples,)" in source
