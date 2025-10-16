from typing import NamedTuple, Optional
from itertools import starmap
from functools import partial
import pytest
import numpy as np

from drillcore_transformations import transformations, utils

# Partial for Lope 202510 measurements
lope_202510_measurement = partial(
    transformations.Measurement, drillcore_plunge=-85, drillcore_trend=213
)

# Partial for Synt 2025 measurements
synt_2025_measurement = partial(
    transformations.Measurement, drillcore_trend=260, drillcore_plunge=-40
)


class MeasurementValidation(NamedTuple):
    measurement: transformations.Measurement
    compass_dip: float
    compass_dir: float
    compass_plunge: Optional[float] = None
    compass_trend: Optional[float] = None
    error_margin: float = 20


measurement_validations_lope_202510 = (
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=14, beta=42, gamma=213),
        compass_dip=82,
        compass_dir=75,
        compass_plunge=22,
        compass_trend=176,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=18, beta=23),
        compass_dip=83,
        compass_dir=57,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=14, beta=175),
        compass_dip=73,
        compass_dir=216,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=14, beta=167),
        compass_dip=68,
        compass_dir=207,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=15, beta=168),
        compass_dip=73,
        compass_dir=216,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=21, beta=159),
        compass_dip=61,
        compass_dir=213,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=25, beta=45),
        compass_dip=68,
        compass_dir=78,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=17, beta=155),
        compass_dip=64,
        compass_dir=200,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=12, beta=95),
        compass_dip=88,
        compass_dir=101,
        # (Compass) measurement error?
        error_margin=30.0,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=63, beta=201),
        compass_dip=23,
        compass_dir=245,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=65, beta=144),
        compass_dip=21,
        compass_dir=189,
    ),
    MeasurementValidation(
        measurement=lope_202510_measurement(alpha=62, beta=132),
        compass_dip=19,
        compass_dir=166,
    ),
)

measurement_validations_synt_2025 = (
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=90, beta=0),
        compass_dip=40,
        compass_dir=80,
    ),
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=75, beta=312),
        compass_dip=45,
        compass_dir=66,
    ),
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=60, beta=170),
        compass_dip=12,
        compass_dir=90,
    ),
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=50, beta=20),
        compass_dip=80,
        compass_dir=92,
    ),
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=35, beta=170),
        compass_dip=18,
        compass_dir=242,
    ),
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=45, beta=135),
        compass_dip=43,
        compass_dir=166,
    ),
    MeasurementValidation(
        measurement=synt_2025_measurement(alpha=80, beta=200),
        compass_dip=32,
        compass_dir=65,
    ),
)

lope_params = starmap(
    partial(pytest.param, id="lope_202510_measurement"),
    measurement_validations_lope_202510,
)

synt_params = starmap(
    partial(pytest.param, id="synt_2025_measurement"),
    measurement_validations_synt_2025,
)


@pytest.mark.parametrize(
    "measurement,compass_dip,compass_dir,compass_plunge,compass_trend,error_margin",
    [*lope_params, *synt_params],
)
def test_measurement(
    measurement, compass_dip, compass_dir, compass_plunge, compass_trend, error_margin
):
    assert 0 <= compass_dir <= 360
    assert 0 <= compass_dip <= 90
    result = transformations.transform(
        measurement.alpha,
        measurement.beta,
        measurement.drillcore_trend,
        measurement.drillcore_plunge,
        measurement.gamma,
    )
    diff_two_planes = utils.calc_difference_between_two_planes(
        dip_first=result[0],
        dir_first=result[1],
        dip_second=compass_dip,
        dir_second=compass_dir,
    )

    print("Planar measurements:")
    print(
        tuple(map(int, result[:2])),
        [
            compass_dip,
            compass_dir,
        ],
    )
    print(f"Difference in angle between planes, in degrees: {diff_two_planes}")
    if diff_two_planes < error_margin:
        print("Angle between planes within error margin")
    else:
        raise ValueError(
            f"Angle between planes ({diff_two_planes}) not within error margin"
        )

    if measurement.gamma is None:
        print("No gamma measurement to test")
        return
    print("Linear measurements:")
    result_plunge = result[2]
    result_trend = result[3]
    assert result_plunge is not None
    assert result_trend is not None
    print(
        (int(result_plunge), int(result_trend)),
        [
            compass_plunge,
            compass_trend,
        ],
    )
    vec_first = transformations.vector_from_dip_and_dir(
        dip=result_plunge,
        dip_dir=result_trend,
    )
    vec_second = transformations.vector_from_dip_and_dir(
        dip=compass_plunge,
        dip_dir=compass_trend,
    )
    diff = np.rad2deg(np.arccos(np.dot(vec_first, vec_second)))
    if diff < error_margin:
        print("Angle between vectors within error margin")
    else:
        raise ValueError(f"Angle between vectors ({diff}) not within error margin")
    # assert
    # diff = diff if diff <= 90 else 180 - diff
    # assert np.isclose(compass_plunge, result_plunge), diff
    # assert np.isclose(compass_trend, result_trend), diff
