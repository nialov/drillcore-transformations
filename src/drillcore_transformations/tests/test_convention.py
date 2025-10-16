from typing import NamedTuple, Optional
from itertools import starmap
import pytest
import numpy as np

from drillcore_transformations import transformations, utils


class MeasurementValidation(NamedTuple):
    measurement: transformations.Measurement
    compass_dip: float
    compass_dir: float
    compass_plunge: Optional[float] = None
    compass_trend: Optional[float] = None
    error_margin: float = 25


measurement_validations_lope_202510 = (
    MeasurementValidation(
        measurement=transformations.Measurement(14, 42, 213, -85, 53),
        compass_dip=82,
        compass_dir=75,
        compass_plunge=22,
        compass_trend=176,
    ),
    MeasurementValidation(
        measurement=transformations.Measurement(
            alpha=18, beta=23, drillcore_plunge=-85, drillcore_trend=53
        ),
        compass_dip=83,
        compass_dir=57,
    ),
    MeasurementValidation(
        measurement=transformations.Measurement(
            alpha=14, beta=175, drillcore_plunge=-85, drillcore_trend=53
        ),
        compass_dip=73,
        compass_dir=216,
    ),
    MeasurementValidation(
        measurement=transformations.Measurement(
            alpha=14, beta=167, drillcore_plunge=-85, drillcore_trend=53
        ),
        compass_dip=68,
        compass_dir=207,
    ),
    MeasurementValidation(
        measurement=transformations.Measurement(
            alpha=15, beta=166, drillcore_plunge=-85, drillcore_trend=53
        ),
        compass_dip=73,
        compass_dir=216,
    ),
)
# alfa beta dip dir
# 25	45				68	78
# 4	355
# 18	23				83	57
# 14	175				73	216
# 14	167				68	207
# 15	168				73	216
# 21	159				61	213
# 17	155				64	200
# 12	95				88	101
# 63	201				23	245
# 65	144				21	189
# 62	132				19	166


@pytest.mark.parametrize(
    "measurement,compass_dip,compass_dir,compass_plunge,compass_trend,error_margin",
    starmap(pytest.param, measurement_validations_lope_202510),
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

    if measurement.gamma is not None:
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
        assert np.isclose(compass_plunge, result_plunge)
        assert np.isclose(compass_trend, result_trend)
