"""Sentinel-2 band selections and the normalisation each arm uses.

Each foundation model gets the bands and statistics it was pretrained with;
the U-Nets get statistics of the Sen1Floods11 training split. All values
are in reflectance x 1e4, the scale of the Sen1Floods11 S2Hand chips.
"""

from __future__ import annotations

from dataclasses import dataclass

# Band order of the 13-band Sen1Floods11 S2Hand chips (Sentinel-2 L1C).
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")

# Central wavelengths in micrometres.
S2_WAVELENGTHS = {
    "B01": 0.443, "B02": 0.493, "B03": 0.560, "B04": 0.665, "B05": 0.704, "B06": 0.740, "B07": 0.783,
    "B08": 0.842, "B8A": 0.865, "B09": 0.945, "B10": 1.375, "B11": 1.610, "B12": 2.190,
}

# Produced by scripts/compute_band_stats.py on the train split: labelled pixels only,
# 57,303,407 pixels.
SEN1FLOODS11_STATS: dict[str, tuple[float, float]] = {
    "B01": (1500.0, 328.4),
    "B02": (1253.1, 352.1),
    "B03": (1227.3, 374.6),
    "B04": (1061.8, 532.1),
    "B05": (1335.3, 451.9),
    "B06": (2298.3, 686.9),
    "B07": (2769.8, 882.4),
    "B08": (2557.8, 851.1),
    "B8A": (3007.0, 1010.2),
    "B09": (459.4, 294.0),
    "B10": (62.3, 146.8),
    "B11": (1934.6, 808.8),
    "B12": (1069.9, 603.6),
}


@dataclass(frozen=True)
class BandSpec:
    name: str
    bands: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.bands) == len(self.mean) == len(self.std)):
            raise ValueError(f"{self.name}: bands, mean and std must have the same length")
        unknown = set(self.bands) - set(S2_BANDS)
        if unknown:
            raise ValueError(f"{self.name}: unknown bands {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.bands)

    @property
    def indices(self) -> list[int]:
        """Positions of these bands in a 13-band S2Hand chip."""
        return [S2_BANDS.index(b) for b in self.bands]

    @property
    def wavelengths(self) -> list[float]:
        return [S2_WAVELENGTHS[b] for b in self.bands]


def _sen1floods11(name: str, bands: tuple[str, ...]) -> BandSpec:
    return BandSpec(
        name, bands, tuple(SEN1FLOODS11_STATS[b][0] for b in bands), tuple(SEN1FLOODS11_STATS[b][1] for b in bands)
    )


# Prithvi-EO-2.0's six HLS bands (Blue, Green, Red, NIR narrow, SWIR 1, SWIR 2).
_PRITHVI_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")

S1F11_13 = _sen1floods11("s1f11_13", S2_BANDS)
S1F11_6 = _sen1floods11("s1f11_6", _PRITHVI_BANDS)

# ibm-nasa-geospatial/Prithvi-EO-2.0-300M config.json (pretrained_cfg mean/std).
PRITHVI_6 = BandSpec(
    "prithvi6",
    _PRITHVI_BANDS,
    (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0),
    (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0),
)

# Clay v1.5 configs/metadata.yaml, sentinel-2-l2a.
CLAY_10 = BandSpec(
    "clay10",
    ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
    (1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0),
    (1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0),
)

BAND_SPECS = {s.name: s for s in (S1F11_13, S1F11_6, PRITHVI_6, CLAY_10)}


def get_band_spec(name: str) -> BandSpec:
    try:
        return BAND_SPECS[name]
    except KeyError:
        raise KeyError(f"unknown band spec {name!r}; choose from {sorted(BAND_SPECS)}") from None
