from pathlib import Path

import numpy as np
import pytest

from roadblock_localization.ipm_ground_projector import IPMGroundProjector


CALIBRATION = Path(__file__).parents[1] / "config" / "ipm_calibration.yaml"


def test_calibration_loads_and_projects_finite_pixels():
    projector = IPMGroundProjector(CALIBRATION)
    points = ((320.0, 350.0), (200.0, 250.0), (400.0, 250.0))
    ground = projector.pixel_to_ground_many(points)
    assert ground.shape == (3, 2)
    assert np.isfinite(ground).all()
    for index, point in enumerate(points):
        assert np.allclose(projector.pixel_to_ground(*point), ground[index], rtol=0.0, atol=1e-12)


def test_projector_rejects_nonfinite_and_outside_pixels():
    projector = IPMGroundProjector(CALIBRATION)
    with pytest.raises(ValueError, match="finite"):
        projector.pixel_to_ground(float("nan"), 100.0)
    with pytest.raises(ValueError, match="outside"):
        projector.pixel_to_ground(-1.0, 100.0)
