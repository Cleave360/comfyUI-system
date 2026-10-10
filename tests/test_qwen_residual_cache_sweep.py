import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_qwen_residual_cache.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qwen_cache_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sequence_has_determinism_control_and_alternating_pair_order():
    module = load_module()
    sequence = module.sequence_for((10, 11, 12))

    assert [(item["mode"], item["seed"]) for item in sequence] == [
        ("stock", 9), ("stock", 9),
        ("stock", 10), ("cache", 10),
        ("cache", 11), ("stock", 11),
        ("stock", 12), ("cache", 12),
    ]


def test_promotion_criteria_are_predeclared():
    module = load_module()

    assert module.CRITERIA == {
        "control_rgb_mae_max": 0.01,
        "candidate_rgb_mae_max_each": 4.0,
        "candidate_psnr_db_min_each": 32.0,
        "paired_median_seconds_saved_min": 2.0,
        "cache_skips_expected_per_run": 2,
    }


def test_cache_bust_is_a_latent_passthrough_upstream_of_sampler():
    module = load_module()
    workflow = {
        "latent": {"class_type": "EmptyLatentImage", "inputs": {}},
        "sampler": {"class_type": "KSampler", "inputs": {"latent_image": ["latent", 0]}},
    }

    module.add_cache_bust(workflow, 7)

    assert workflow["kindred_cache_bust"] == {
        "class_type": "KindredCacheBustLatent",
        "inputs": {"samples": ["latent", 0], "nonce": 7},
    }
    assert workflow["sampler"]["inputs"]["latent_image"] == ["kindred_cache_bust", 0]


def test_sweep_has_an_explicit_cooldown_option():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--cooldown-seconds", type=float, default=60.0)' in source
    assert "time.sleep(args.cooldown_seconds)" in source
