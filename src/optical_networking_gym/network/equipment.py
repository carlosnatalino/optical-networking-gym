"""Equipment library in the GNPy ``eqpt_config.json`` format.

The library describes *types* of network elements (amplifiers, ROADMs,
transceivers) and is read from the same JSON schema used by GNPy
[Curri_2022_GNPyModelPhysical], so operators' or GNPy's example equipment files
can be reused. Only the fields needed by the gym's closed-form QoT model are
interpreted:

* ``Edfa``: ``type_variety``, ``type_def``, ``gain_flatmax``, ``gain_min``,
  ``p_max`` and the noise-figure (NF) model. Three GNPy NF models are ported:
  ``fixed_gain`` (``nf0``), ``variable_gain`` (two-coil model estimated from
  ``nf_min``/``nf_max``) and ``advanced_model`` (polynomial ``nf_fit_coeff`` in
  the gain offset from ``gain_flatmax``). Any amplifier may reference an
  ``advanced_config_from_json`` file whose ``gain_ripple`` profile (dB over
  ``f_min``..``f_max``) models the wavelength-dependent gain ripple of EDFAs
  [Mahajan_2020_ModelingEDFAGain].
* ``Span``: default connector losses ``con_in``/``con_out`` (dB).
* ``Roadm``: ``add_drop_osnr`` (dB) of the add/drop structure.
* ``SI`` and ``Transceiver``: the transceiver ``tx_osnr`` (dB).

Amplifier types with other ``type_def`` values (OpenROADM, dual-stage, Raman)
are parsed but not usable for NF computation; asking for their NF raises.

The NF models follow GNPy's implementation (``gnpy/core/elements.py``
``Edfa._nf`` and ``gnpy/core/science_utils.py`` ``estimate_nf_model``, BSD
3-Clause License, Telecom Infra Project).

References are listed at the end of the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "AmplifierType",
    "EquipmentLibrary",
    "GainRipple",
    "db_to_linear",
    "estimate_two_coil_nf_model",
    "linear_to_db",
]

_SUPPORTED_NF_MODELS = frozenset({"fixed_gain", "variable_gain", "advanced_model"})


def db_to_linear(value_db: float) -> float:
    """Convert dB to a linear ratio."""
    return float(10.0 ** (value_db / 10.0))


def linear_to_db(value: float) -> float:
    """Convert a linear ratio to dB."""
    return float(10.0 * math.log10(value))


@dataclass(frozen=True, slots=True)
class GainRipple:
    """Wavelength-dependent gain deviation of an amplifier around its target gain.

    Attributes:
        frequencies_hz: Increasing frequency grid (Hz).
        values_db: Gain deviation (dB) at each grid frequency.
    """

    frequencies_hz: tuple[float, ...]
    values_db: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.frequencies_hz) != len(self.values_db) or len(self.values_db) < 2:
            raise ValueError("gain ripple needs >= 2 matching frequency/value samples")
        if any(b <= a for a, b in zip(self.frequencies_hz, self.frequencies_hz[1:])):
            raise ValueError("gain ripple frequencies must be strictly increasing")

    @classmethod
    def uniform(cls, f_min_hz: float, f_max_hz: float, values_db: list[float] | tuple[float, ...]) -> "GainRipple":
        """Build a ripple sampled uniformly between ``f_min_hz`` and ``f_max_hz``."""
        grid = np.linspace(f_min_hz, f_max_hz, len(values_db))
        return cls(tuple(float(f) for f in grid), tuple(float(v) for v in values_db))

    def at(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Gain deviation (dB) at the given frequencies; edge values outside the grid."""
        return np.interp(
            np.asarray(frequencies_hz, dtype=np.float64),
            np.asarray(self.frequencies_hz),
            np.asarray(self.values_db),
        )

    def transformed(self, *, scale: float = 1.0, shift_hz: float = 0.0) -> "GainRipple":
        """Return ``scale * ripple(f - shift_hz)``.

        Random shifts and amplitude scalings of a measured profile are the way
        per-amplifier ripple diversity is generated in
        [Mahajan_2020_ModelingEDFAGain].
        """
        return GainRipple(
            tuple(f + shift_hz for f in self.frequencies_hz),
            tuple(scale * v for v in self.values_db),
        )


def estimate_two_coil_nf_model(
    type_variety: str, gain_min: float, gain_max: float, nf_min: float, nf_max: float
) -> tuple[float, float, float]:
    """Two-coil NF model of a variable-gain EDFA (port of GNPy ``estimate_nf_model``).

    Solves ``nf_{min,max} = nf1 + nf2 / g1a_{min,max}`` for the first- and
    second-coil noise figures ``nf1``, ``nf2`` (dB) and the inter-coil power
    difference ``delta_p`` (dB).

    Returns:
        ``(nf1, nf2, delta_p)``.

    Raises:
        ValueError: If the inputs do not describe a physical two-coil amplifier.
    """
    if nf_min < -10 or nf_max < -10:
        raise ValueError(f"invalid nf_min/nf_max for amplifier {type_variety}")
    delta_p = 5.0
    g1a_min = gain_min - (gain_max - gain_min) - delta_p
    g1a_max = gain_max - delta_p
    nf2 = linear_to_db(
        (db_to_linear(nf_min) - db_to_linear(nf_max))
        / (1.0 / db_to_linear(g1a_max) - 1.0 / db_to_linear(g1a_min))
    )
    nf1 = linear_to_db(db_to_linear(nf_min) - db_to_linear(nf2) / db_to_linear(g1a_max))
    if nf1 < 4:
        raise ValueError(f"first coil NF too low ({nf1:.2f} dB) for amplifier {type_variety}")
    if not nf1 + 0.3 < nf2 < nf1 + 2:
        nf2 = float(np.clip(nf2, nf1 + 0.3, nf1 + 2))
        g1a_max = linear_to_db(db_to_linear(nf2) / (db_to_linear(nf_min) - db_to_linear(nf1)))
        delta_p = gain_max - g1a_max
        g1a_min = gain_min - (gain_max - gain_min) - delta_p
        if not 1 < delta_p < 11:
            raise ValueError(f"invalid inter-coil delta_p ({delta_p:.2f} dB) for {type_variety}")
    calc_nf_min = linear_to_db(db_to_linear(nf1) + db_to_linear(nf2) / db_to_linear(g1a_max))
    calc_nf_max = linear_to_db(db_to_linear(nf1) + db_to_linear(nf2) / db_to_linear(g1a_min))
    if not (math.isclose(nf_min, calc_nf_min, abs_tol=0.01) and math.isclose(nf_max, calc_nf_max, abs_tol=0.01)):
        raise ValueError(f"inconsistent two-coil NF model for amplifier {type_variety}")
    return nf1, nf2, delta_p


@dataclass(frozen=True, slots=True)
class AmplifierType:
    """An EDFA type from the ``Edfa`` list of the equipment file."""

    type_variety: str
    type_def: str
    gain_flatmax: float
    gain_min: float
    p_max: float
    nf0: float | None = None
    nf_min: float | None = None
    nf_max: float | None = None
    nf_fit_coeff: tuple[float, ...] | None = None
    gain_ripple: GainRipple | None = None
    _two_coil: tuple[float, float, float] | None = field(default=None, repr=False, compare=False)

    @property
    def supports_nf(self) -> bool:
        """Whether :meth:`noise_figure_db` is available for this type."""
        return self.type_def in _SUPPORTED_NF_MODELS

    def noise_figure_db(self, gain_db: float) -> float:
        """Average NF (dB) at a given operating gain (port of GNPy ``Edfa._nf``).

        Gains below ``gain_min`` are reached with an input pad (VOA) whose loss
        adds to the NF, as in GNPy.
        """
        pad = max(self.gain_min - gain_db, 0.0)
        gain_target = gain_db + pad
        dg = max(self.gain_flatmax - gain_target, 0.0)
        if self.type_def == "fixed_gain":
            if self.nf0 is None:
                raise ValueError(f"fixed_gain amplifier {self.type_variety} lacks nf0")
            nf_avg = self.nf0
        elif self.type_def == "variable_gain":
            if self._two_coil is None:
                raise ValueError(f"variable_gain amplifier {self.type_variety} lacks nf_min/nf_max")
            nf1, nf2, delta_p = self._two_coil
            g1a = gain_target - delta_p - dg
            nf_avg = linear_to_db(db_to_linear(nf1) + db_to_linear(nf2) / db_to_linear(g1a))
        elif self.type_def == "advanced_model":
            if self.nf_fit_coeff is None:
                raise ValueError(f"advanced_model amplifier {self.type_variety} lacks nf_fit_coeff")
            nf_avg = float(np.polyval(self.nf_fit_coeff, -dg))
        else:
            raise ValueError(f"NF model '{self.type_def}' of {self.type_variety} is not supported")
        return float(nf_avg + pad)


@dataclass(frozen=True, slots=True)
class EquipmentLibrary:
    """Parsed equipment file (see module docstring for the supported fields)."""

    amplifiers: Mapping[str, AmplifierType]
    span_con_in_db: float = 0.0
    span_con_out_db: float = 0.0
    roadm_add_drop_osnr_db: float | None = None
    transceiver_tx_osnr_db: float | None = None
    source: Path | None = None

    def amplifier(self, type_variety: str) -> AmplifierType:
        """Return an amplifier type by name."""
        try:
            return self.amplifiers[type_variety]
        except KeyError as exc:
            raise KeyError(f"amplifier type '{type_variety}' not in equipment library") from exc

    @classmethod
    def from_json(cls, file_path: str | Path) -> "EquipmentLibrary":
        """Read a GNPy-format ``eqpt_config.json``.

        ``advanced_config_from_json`` paths are resolved relative to the
        equipment file.
        """
        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(payload, base_dir=path.parent, source=path)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
        source: Path | None = None,
    ) -> "EquipmentLibrary":
        """Build the library from an already-parsed equipment document."""
        amplifiers: dict[str, AmplifierType] = {}
        for entry in payload.get("Edfa", ()):
            amplifier = _parse_amplifier(entry, base_dir)
            if amplifier is not None:
                amplifiers[amplifier.type_variety] = amplifier
        span = _first(payload.get("Span"))
        roadm = _first(payload.get("Roadm"))
        si = _first(payload.get("SI"))
        return cls(
            amplifiers=amplifiers,
            span_con_in_db=float(span.get("con_in", 0.0)),
            span_con_out_db=float(span.get("con_out", 0.0)),
            roadm_add_drop_osnr_db=_optional_float(roadm.get("add_drop_osnr")),
            transceiver_tx_osnr_db=_optional_float(si.get("tx_osnr")),
            source=source,
        )


def _first(entries: Any) -> Mapping[str, Any]:
    if isinstance(entries, list) and entries:
        return entries[0]
    if isinstance(entries, Mapping):
        return entries
    return {}


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _parse_amplifier(entry: Mapping[str, Any], base_dir: Path | None) -> AmplifierType | None:
    if "type_variety" not in entry or "gain_flatmax" not in entry:
        return None  # dual-stage/composite entries are not single amplifiers
    type_def = str(entry.get("type_def", "variable_gain"))
    gain_ripple = None
    nf_fit_coeff = entry.get("nf_fit_coeff")
    advanced_ref = entry.get("advanced_config_from_json")
    if advanced_ref:
        advanced_path = Path(advanced_ref)
        if not advanced_path.is_absolute() and base_dir is not None:
            advanced_path = base_dir / advanced_path
        advanced = json.loads(advanced_path.read_text(encoding="utf-8"))
        ripple_values = advanced.get("gain_ripple")
        if isinstance(ripple_values, list) and len(ripple_values) >= 2:
            gain_ripple = GainRipple.uniform(
                float(advanced["f_min"]), float(advanced["f_max"]), ripple_values
            )
        nf_fit_coeff = nf_fit_coeff or advanced.get("nf_fit_coeff")
    two_coil = None
    if type_def == "variable_gain" and "nf_min" in entry and "nf_max" in entry:
        two_coil = estimate_two_coil_nf_model(
            str(entry["type_variety"]),
            float(entry.get("gain_min", 0.0)),
            float(entry["gain_flatmax"]),
            float(entry["nf_min"]),
            float(entry["nf_max"]),
        )
    return AmplifierType(
        type_variety=str(entry["type_variety"]),
        type_def=type_def,
        gain_flatmax=float(entry["gain_flatmax"]),
        gain_min=float(entry.get("gain_min", 0.0)),
        p_max=float(entry.get("p_max", float("inf"))),
        nf0=_optional_float(entry.get("nf0")),
        nf_min=_optional_float(entry.get("nf_min")),
        nf_max=_optional_float(entry.get("nf_max")),
        nf_fit_coeff=tuple(float(c) for c in nf_fit_coeff) if nf_fit_coeff else None,
        gain_ripple=gain_ripple,
        _two_coil=two_coil,
    )


# References
# [Curri_2022_GNPyModelPhysical] V. Curri, "GNPy Model of the Physical Layer for Open
#     and Disaggregated Optical Networking [Invited]," Journal of Optical Communications
#     and Networking, vol. 14, no. 6, pp. C92-C104, Jun. 2022, doi: 10.1364/JOCN.452868.
# [Mahajan_2020_ModelingEDFAGain] A. Mahajan, K. Christodoulopoulos, R. Martinez,
#     S. Spadaro, and R. Munoz, "Modeling EDFA Gain Ripple and Filter Penalties With
#     Machine Learning for Accurate QoT Estimation," Journal of Lightwave Technology,
#     vol. 38, no. 9, pp. 2616-2629, May 2020, doi: 10.1109/JLT.2020.2975081.
