import pytest

from drillcore_transformations import transformations


measurement_1 = transformations.Measurement(14, 42, 213, -85, 53)

measurement_1_compass = (82, 75, 22, 176)


@pytest.mark.parametrize(
    "alpha, beta, drillcore_trend, drillcore_plunge, gamma, plane_dip, plane_dir, gamma_plunge, gamma_trend, name",
    [[*measurement_1, *measurement_1_compass, "nn"]],
)
def test_measurement(
    alpha,
    beta,
    drillcore_trend,
    drillcore_plunge,
    gamma,
    plane_dip,
    plane_dir,
    gamma_plunge,
    gamma_trend,
):
    result = transformations.transform_with_gamma(
        alpha, beta, drillcore_trend, drillcore_plunge, gamma
    )

    try:
        assert result[0] == plane_dip, "Dip is incorrect"
        assert result[1] == plane_dir, "Dir is incorrect"
        assert result[2] == gamma_plunge, "Plunge is incorrect"
        assert result[3] == gamma_trend, "Trend is incorrect"

    except Exception:
        print(
            result,
            [
                plane_dip,
                plane_dir,
                gamma_plunge,
                gamma_trend,
            ],
        )
        raise
