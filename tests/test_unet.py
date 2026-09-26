"""The flood segmentation model's contract and shape arithmetic.

No weights exist and torch is not a hard dependency, so these tests cover the
part that can be checked anywhere: the band contract, the configuration
validation, and the shape planning. That is not a consolation prize — most of
what goes wrong in a segmentation pipeline goes wrong in the shapes and the band
ordering, and those failures surface twenty minutes into a GPU run if nothing
catches them first.

The tests that need torch skip rather than fail, so the suite means the same
thing on a laptop, in CI and on a training box.
"""

from __future__ import annotations

import pytest

from satai.errors import ValidationError
from satai.ml.unet import (
    BAND_PRESETS,
    FloodUNetConfig,
    build_model,
    plan_shapes,
    torch_available,
)


class TestBandContract:
    def test_every_preset_starts_from_sar(self) -> None:
        """SAR is the anchor modality: it is the only one that sees through
        cloud, and a preset without it is not a flood model this project would
        defend."""
        for name, bands in BAND_PRESETS.items():
            assert bands[0] == "vv_db", f"{name} does not lead with VV"

    def test_presets_have_no_duplicate_bands(self) -> None:
        for name, bands in BAND_PRESETS.items():
            assert len(set(bands)) == len(bands), name

    def test_the_full_preset_is_a_superset_of_the_others(self) -> None:
        """C2 compares degraded stacks against the full one. If a preset carried
        a band the full stack lacked, the comparison would not be an ablation."""
        full = set(BAND_PRESETS["full"])
        for name, bands in BAND_PRESETS.items():
            assert set(bands) <= full, f"{name} has bands absent from 'full'"

    def test_an_unknown_preset_names_the_known_ones(self) -> None:
        with pytest.raises(ValidationError, match="known presets"):
            FloodUNetConfig.from_preset("sar_plus_vibes")

    def test_the_band_list_travels_into_the_checkpoint_description(self) -> None:
        """A model served with a different band order than it was trained on
        produces confident nonsense that nothing downstream can detect."""
        config = FloodUNetConfig.from_preset("sar_dem")
        described = config.describe()
        assert described["bands"] == list(BAND_PRESETS["sar_dem"])
        assert described["in_channels"] == 5


class TestConfigValidation:
    def test_a_chip_the_decoder_cannot_restore_is_rejected(self) -> None:
        """500 is not divisible by 2**5, so the output mask would come back a
        different shape from the label and the loss would silently broadcast."""
        with pytest.raises(ValidationError, match="not divisible"):
            FloodUNetConfig(chip_px=500, depth=5)

    def test_a_valid_chip_size_is_accepted(self) -> None:
        assert FloodUNetConfig(chip_px=512, depth=5).chip_px == 512
        assert FloodUNetConfig(chip_px=256, depth=4).chip_px == 256

    def test_duplicate_bands_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate bands"):
            FloodUNetConfig(bands=("vv_db", "vv_db"))

    def test_an_empty_band_stack_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one input band"):
            FloodUNetConfig(bands=())


class TestPretrainingHonesty:
    def test_three_channels_can_reuse_the_pretrained_first_layer(self) -> None:
        config = FloodUNetConfig.from_preset("sar_ratio")
        assert config.in_channels == 3
        assert config.pretrained_first_layer_reusable is True

    def test_a_wider_stack_cannot_and_says_so(self) -> None:
        """ "Pretrained" is not free for a 7-band stack: the first convolution is
        replaced, so the claim is qualified rather than repeated."""
        config = FloodUNetConfig.from_preset("full")
        assert config.pretrained_first_layer_reusable is False

        caveats = " ".join(config.describe()["caveats"])
        assert "re-initialised" in caveats
        assert "second block onward" in caveats

    def test_the_urban_failure_mode_is_always_carried(self) -> None:
        """C3 measures exactly this degradation; the caveat must not depend on
        which preset was chosen."""
        for name in BAND_PRESETS:
            caveats = " ".join(FloodUNetConfig.from_preset(name).describe()["caveats"])
            assert "double-bounce" in caveats
            assert "Sen1Floods11" in caveats


class TestShapePlanning:
    def test_the_decoder_restores_the_input_resolution(self) -> None:
        shapes = plan_shapes(FloodUNetConfig(chip_px=512, depth=5))
        assert shapes["input"][-2:] == shapes["output"][-2:] == (512, 512)

    def test_channel_depth_follows_the_band_count(self) -> None:
        shapes = plan_shapes(FloodUNetConfig.from_preset("full"), batch=4)
        assert shapes["input"] == (4, 7, 512, 512)

    def test_each_encoder_level_halves_the_resolution(self) -> None:
        shapes = plan_shapes(FloodUNetConfig(chip_px=512, depth=5))
        assert shapes["encoder_0"][-1] == 256
        assert shapes["encoder_4"][-1] == 16

    def test_the_output_carries_one_class_per_configured_class(self) -> None:
        shapes = plan_shapes(FloodUNetConfig(n_classes=1), batch=2)
        assert shapes["output"] == (2, 1, 512, 512)


class TestBuild:
    def test_building_without_torch_explains_rather_than_tracebacks(self) -> None:
        """The common way to hit this is running training code on the serving
        host, which deserves a sentence rather than an ImportError."""
        if torch_available():
            pytest.skip("torch is installed; the no-torch path cannot be exercised")
        with pytest.raises(ValidationError, match="PyTorch is not installed"):
            build_model(FloodUNetConfig())

    def test_the_model_accepts_a_chip_and_returns_a_mask(self) -> None:
        if not torch_available():
            pytest.skip("torch is not installed in this environment")
        import torch

        config = FloodUNetConfig.from_preset("sar_ratio", chip_px=64, depth=5)
        model = build_model(FloodUNetConfig(**{**config.__dict__, "encoder_weights": None}))
        with torch.no_grad():
            out = model(torch.zeros(1, config.in_channels, 64, 64))
        assert tuple(out.shape) == (1, 1, 64, 64)
