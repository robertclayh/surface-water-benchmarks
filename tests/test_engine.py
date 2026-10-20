import json

from swbench.config import apply_overrides, deep_merge, load_config
from swbench.train.engine import run, warmup_cosine


def test_schedule():
    assert warmup_cosine(0, 100, 10) == 0.1
    assert warmup_cosine(10, 100, 10) == 1.0
    assert warmup_cosine(100, 100, 10) < 1e-6


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/arms/prithvi_finetune.yaml")
    assert cfg["train"]["steps"] == 2000 and cfg["train"]["encoder_lr"] == 2e-5
    cfg = apply_overrides(cfg, ["train.steps=5", "label_budget=8", "data.val_chips=null"])
    assert cfg["train"]["steps"] == 5 and cfg["label_budget"] == 8 and cfg["data"]["val_chips"] is None
    cfg = apply_overrides(cfg, ["train.lr=3e-4", "train.encoder_lr=1E-5", "eval.splits=[valid]", "name=1e3x"])
    assert cfg["train"]["lr"] == 3e-4 and cfg["train"]["encoder_lr"] == 1e-5
    assert cfg["eval"]["splits"] == ["valid"] and cfg["name"] == "1e3x"
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def _cfg(root, arch="unet", bands="s1f11_6", protocol="scratch", kwargs=None):
    return {
        "name": f"{arch}_{protocol}",
        "seed": 0,
        "label_budget": 4,
        "device": "cpu",
        "data": {"root": str(root), "crop_size": 32, "workers": 0, "val_chips": None},
        "train": {"steps": 6, "batch_size": 2, "lr": 1e-3, "eval_every": 3, "warmup_steps": 1},
        "eval": {"tile": 32, "overlap": 8, "batch_size": 2, "bootstrap": 50},
        "model": {"arch": arch, "bands": bands, "protocol": protocol, "kwargs": kwargs or {"base_width": 8}},
    }


def test_run_end_to_end(fake_sen1floods11, tmp_path):
    row = run(_cfg(fake_sen1floods11), tmp_path / "run")
    assert row["label_budget"] == 4
    for split in ("valid", "test", "bolivia"):
        assert 0.0 <= row[f"{split}_iou_water"] <= 1.0
        assert row[f"{split}_iou_water_lo"] <= row[f"{split}_iou_water_hi"]
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["arm"] == "unet_scratch"
    assert len((tmp_path / "run" / "train_chips.txt").read_text().split()) == 4


def test_run_frozen_vit(fake_sen1floods11, tmp_path):
    kw = {"embed_dim": 32, "depth": 2, "heads": 2, "patch_size": 16, "decoder_channels": 16}
    row = run(_cfg(fake_sen1floods11, "simple_vit", "prithvi6", "frozen", kw), tmp_path / "run")
    assert row["params_trainable"] < row["params_total"]
