from __future__ import annotations

import pytest

from optical_networking_gym_v2 import Modulation, QoTResult, ServiceQoTUpdate


def test_modulation_validates_spectral_efficiency() -> None:
    with pytest.raises(ValueError, match="spectral_efficiency"):
        Modulation(
            name="INVALID",
            maximum_length=1000.0,
            spectral_efficiency=0,
            minimum_osnr=10.0,
            inband_xt=-17.0,
        )


def test_qot_result_uses_modulation_threshold() -> None:
    modulation = Modulation(
        name="QPSK",
        maximum_length=200_000.0,
        spectral_efficiency=2,
        minimum_osnr=6.72,
        inband_xt=-17.0,
    )
    result = QoTResult(osnr=8.0, ase=10.0, nli=12.0, meets_threshold=True)
    update = ServiceQoTUpdate(service_id=4, osnr=8.0, ase=10.0, nli=12.0)

    assert modulation.minimum_osnr == pytest.approx(6.72)
    assert result.meets_threshold is True
    assert update.service_id == 4
