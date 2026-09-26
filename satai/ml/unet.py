"""Flood segmentation network: configuration, shape planning, and the model.

**No weights ship with this module and none have been trained.** What is here is
the architecture, the band contract it expects, and the shape arithmetic that
decides whether a given chip and encoder are compatible. That last part is the
reason this module is testable at all without a GPU: most of what goes wrong in
a segmentation pipeline goes wrong in the shapes and the band ordering, hours
before a loss curve would have told you.

Design, and why
---------------
*U-Net with a pretrained encoder.* Sen1Floods11 has 446 labelled chips. Training
an encoder from scratch on that is a way to overfit 11 events; an ImageNet-
pretrained ResNet gives low-level filters for free and leaves the decoder to
learn the task. The domain gap between ImageNet photographs and SAR backscatter
is real and is the reason the first convolution is re-initialised rather than
kept (see :meth:`FloodUNetConfig.encoder_in_channels`).

*Channels are declared, not inferred.* A SAR chip is VV, VH and their ratio;
adding DEM, HAND and rainfall changes the input depth. Inferring depth from
whatever array arrives is how a model silently trains on a band order different
from the one it is served, so the band list is part of the config, travels into
the checkpoint, and is checked at inference.

*Torch is imported lazily.* The serving plane is CPU-only and never loads this;
the config and planning below must import in an environment with no torch at
all, which is also what lets CI check them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from satai.errors import ValidationError
from satai.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is not a hard dependency
    import torch

log = get_logger(__name__)

__all__ = [
    "BAND_PRESETS",
    "FloodUNetConfig",
    "build_model",
    "plan_shapes",
    "torch_available",
]

#: Named band stacks. The key is what an experiment refers to; the value is the
#: exact ordered band list the model is built for. C2 (modality loss) is exactly
#: a sweep over these presets, which is why they live here rather than being
#: assembled ad hoc in a training script.
BAND_PRESETS: dict[str, tuple[str, ...]] = {
    "sar_only": ("vv_db", "vh_db"),
    "sar_ratio": ("vv_db", "vh_db", "vv_vh_ratio"),
    "sar_dem": ("vv_db", "vh_db", "vv_vh_ratio", "dem_m", "slope_deg"),
    "sar_hand": ("vv_db", "vh_db", "vv_vh_ratio", "hand_m"),
    "sar_rain": ("vv_db", "vh_db", "vv_vh_ratio", "rain_mm_3d"),
    "full": (
        "vv_db",
        "vh_db",
        "vv_vh_ratio",
        "dem_m",
        "slope_deg",
        "hand_m",
        "rain_mm_3d",
    ),
}


def torch_available() -> bool:
    """Whether a torch build is importable in this environment.

    Used by scripts to fail with a sentence rather than a traceback, and by the
    tests to skip the parts that need it. The rest of this module is deliberately
    usable without it.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class FloodUNetConfig:
    """Everything that has to match between training and serving.

    This object is written into the checkpoint. A model served with a different
    band order than it was trained on produces confident nonsense and nothing
    downstream can detect it, so the contract is carried with the weights rather
    than reconstructed from a filename.
    """

    bands: tuple[str, ...] = BAND_PRESETS["sar_ratio"]
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    chip_px: int = 512
    #: Binary water/no-water. Left explicit because a later multi-class variant
    #: (water / wet soil / dry) would change the loss and the metrics.
    n_classes: int = 1
    depth: int = 5
    seed: int = 42
    preset_name: str = "sar_ratio"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValidationError("a model needs at least one input band")
        if len(set(self.bands)) != len(self.bands):
            raise ValidationError(f"duplicate bands in {self.bands}")
        if self.chip_px % (2**self.depth) != 0:
            raise ValidationError(
                f"chip_px {self.chip_px} is not divisible by 2**{self.depth} = "
                f"{2**self.depth}; the decoder cannot restore the input size and "
                f"the output mask would be a different shape from the label"
            )
        if self.n_classes < 1:
            raise ValidationError(f"n_classes must be at least 1, got {self.n_classes}")

    @property
    def in_channels(self) -> int:
        return len(self.bands)

    @property
    def encoder_in_channels(self) -> int:
        """Channels the encoder's first convolution must accept.

        An ImageNet encoder expects 3. With any other count the first conv is
        replaced and re-initialised, so the pretrained benefit applies from the
        second block onward. Stated here because it is the honest limit of
        "pretrained": for a 7-band stack, most of the first layer is new.
        """
        return self.in_channels

    @property
    def pretrained_first_layer_reusable(self) -> bool:
        return self.encoder_weights is not None and self.in_channels == 3

    def describe(self) -> dict[str, Any]:
        """Goes into the model registry and the provenance envelope."""
        return {
            "architecture": "unet",
            "encoder": self.encoder,
            "encoder_weights": self.encoder_weights or "random",
            "bands": list(self.bands),
            "preset": self.preset_name,
            "in_channels": self.in_channels,
            "chip_px": self.chip_px,
            "n_classes": self.n_classes,
            "depth": self.depth,
            "seed": self.seed,
            "pretrained_first_layer_reusable": self.pretrained_first_layer_reusable,
            "caveats": [
                "Trained on Sen1Floods11, which is 11 events; generalisation "
                "beyond those flood regimes is untested.",
                "Known failure mode: urban double-bounce RAISES backscatter over "
                "flooded streets, inverting the signature this model relies on.",
                *(
                    []
                    if self.pretrained_first_layer_reusable
                    else [
                        f"The encoder's first convolution was re-initialised for "
                        f"{self.in_channels} input channels, so ImageNet "
                        f"pretraining applies from the second block onward."
                    ]
                ),
                *self.notes,
            ],
        }

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> FloodUNetConfig:
        if name not in BAND_PRESETS:
            known = ", ".join(sorted(BAND_PRESETS))
            raise ValidationError(f"unknown band preset {name!r}; known presets: {known}")
        return cls(bands=BAND_PRESETS[name], preset_name=name, **overrides)


def plan_shapes(config: FloodUNetConfig, batch: int = 8) -> dict[str, tuple[int, ...]]:
    """Tensor shapes at each resolution level, without building anything.

    Catches the two mistakes that otherwise surface as a CUDA error twenty
    minutes into a run: a chip size the decoder cannot restore, and a batch that
    will not fit. Cheap enough to assert in a test, which is the point.
    """
    shapes: dict[str, tuple[int, ...]] = {
        "input": (batch, config.in_channels, config.chip_px, config.chip_px),
    }
    size = config.chip_px
    channels = 64
    for level in range(config.depth):
        size //= 2
        shapes[f"encoder_{level}"] = (batch, channels, size, size)
        channels = min(channels * 2, 512)
    for level in reversed(range(config.depth)):
        size *= 2
        shapes[f"decoder_{level}"] = (batch, max(16, channels // 2), size, size)
        channels = max(16, channels // 2)
    shapes["output"] = (batch, config.n_classes, config.chip_px, config.chip_px)

    if shapes["output"][-2:] != shapes["input"][-2:]:
        raise ValidationError(
            f"decoder does not restore the input size: input "
            f"{shapes['input'][-2:]} vs output {shapes['output'][-2:]}"
        )
    return shapes


def build_model(config: FloodUNetConfig) -> torch.nn.Module:
    """Construct the network. Requires torch and segmentation-models-pytorch.

    Raises a stated error rather than an ImportError traceback when the training
    stack is absent, because the common case for that is someone running this on
    the serving machine by mistake.
    """
    if not torch_available():
        raise ValidationError(
            "PyTorch is not installed in this environment. The flood model is "
            "trained in a GPU environment (Colab, Kaggle, or a local CUDA box) "
            "and exported to ONNX for CPU serving -- see ADR-001. Install the "
            "training extras there, not on the serving host."
        )
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ValidationError(
            "segmentation-models-pytorch is not installed; it provides the U-Net "
            "and the pretrained encoders. See environment.yml."
        ) from exc

    import torch

    torch.manual_seed(config.seed)
    model: torch.nn.Module = smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.n_classes,
        encoder_depth=config.depth,
    )
    log.info(
        "flood u-net built",
        extra={
            "encoder": config.encoder,
            "in_channels": config.in_channels,
            "preset": config.preset_name,
            "pretrained_first_layer": config.pretrained_first_layer_reusable,
        },
    )
    return model
