from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_residual_probe_server.py"


def test_probe_is_measurement_only_and_uses_branch_scoped_history():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"intervention": False' in source
    assert 'state["previous"].get(key)' in source
    assert 'state["previous"][key]' in source
    assert "return original_model_forward(self, *args, **kwargs)" in source
    assert "img_residual = (output_img - input_img).detach()" in source
    assert "txt_residual = (output_txt - input_txt).detach()" in source


def test_probe_report_is_replaced_atomically():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'with_suffix(REPORT_PATH.suffix + ".tmp")' in source
    assert "temporary.replace(REPORT_PATH)" in source
