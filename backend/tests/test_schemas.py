import pytest
from pydantic import ValidationError

from app.schemas.training import (
    TrainingConfig,
    TrainingEstimateRequest,
    estimate_training_time,
)


def _est(**kw):
    base = dict(base_model="meta-llama/Llama-3.2-1B-Instruct", dataset_total_tokens=10_000_000)
    return estimate_training_time(TrainingEstimateRequest(**{**base, **kw}))


def test_config_defaults_are_valid():
    cfg = TrainingConfig(dataset_id="abc")
    assert cfg.use_lora and cfg.load_in_4bit


def test_unsloth_requires_lora():
    with pytest.raises(ValidationError):
        TrainingConfig(dataset_id="abc", use_unsloth=True, use_lora=False, load_in_4bit=False)


def test_4bit_without_lora_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(dataset_id="abc", use_lora=False, load_in_4bit=True)


def test_full_finetune_without_4bit_is_allowed():
    assert TrainingConfig(dataset_id="abc", use_lora=False, load_in_4bit=False)


@pytest.mark.parametrize("field,value", [("epochs", 0), ("learning_rate", 1.0), ("lora_r", 0)])
def test_out_of_range_values_rejected(field, value):
    with pytest.raises(ValidationError):
        TrainingConfig(dataset_id="abc", **{field: value})


def test_estimate_scales_with_epochs():
    assert _est(epochs=4)["estimated_seconds"] > _est(epochs=2)["estimated_seconds"]


def test_bigger_model_is_slower():
    small = _est(base_model="meta-llama/Llama-3.2-1B-Instruct")
    big = _est(base_model="meta-llama/Llama-3.1-8B-Instruct")
    assert big["estimated_seconds"] > small["estimated_seconds"]


def test_faster_gpu_is_faster():
    assert _est(gpu_name="A100-80GB")["estimated_seconds"] < _est(gpu_name="T4-16GB")["estimated_seconds"]


def test_full_finetune_slower_than_qlora():
    qlora = _est()
    full = _est(use_lora=False, load_in_4bit=False)
    assert full["estimated_seconds"] > qlora["estimated_seconds"]
