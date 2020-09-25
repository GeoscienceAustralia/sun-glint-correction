"""
Sun glint approximation based on the Cox and Munk (1954) model, as well as
correction.
"""

import numpy
import numexpr


def cm_sun_glint(
    satellite_view,
    solar_zenith,
    relative_azimuth,
    wind_speed,
    refractive_index=1.34,  # noqa # pylint: disable
):
    """
    Calculate sun glint based on the Cox and Munk (1954) model.

    Calculations are done in single precision (same as the original
    FORTRAN code that this Python function was derived from).

    :param satellite_view:
        The satellite zenith view angle in degrees.

    :param solar_zenith:
        The solar zenith angle in degrees.

    :param relative_azimuth:
        The relative azimuth angle between sun and view direction.

    :param wind_speed:
        The wind speed in m/s. Internally recast as float32.

    :param refractive_index:
        The refractive index of water. Default is 1.34 and internally
        recast as a float32.

    :return:
        A 2D NumPy array of type float32.

    :notes:
        This function is lengthy and potentially could be split into
        multiple functions.
    """

    expr = "relative_azimuth > 180.0"
    angle_mask = numexpr.evaluate(expr)
    relative_azimuth[angle_mask] = relative_azimuth[angle_mask] - 360.0

    # force constants to float32 (reduce memory for array computations)
    ri_ = numpy.float32(refractive_index)  # noqa # pylint: disable=unused-variable
    p5_ = numpy.float32(0.5)  # noqa # pylint: disable=unused-variable
    pi_ = numpy.float32(numpy.pi)  # noqa # pylint: disable=unused-variable
    ws_ = numpy.float32(wind_speed)  # noqa # pylint: disable=unused-variable

    theta_view = numpy.deg2rad(  # noqa # pylint: disable=unused-variable
        satellite_view, dtype="float32"
    )
    theta_sun = numpy.deg2rad(  # noqa # pylint: disable=unused-variable
        solar_zenith, dtype="float32"
    )
    theta_phi = numpy.deg2rad(  # noqa # pylint: disable=unused-variable
        relative_azimuth, dtype="float32"
    )

    expr = "cos(theta_sun) * cos(theta_view) + sin(theta_view) * sin(theta_sun) * cos(theta_phi)"  # noqa # pylint: disable=unused-variable,line-too-long
    cos_psi = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    expr = "sqrt((1.0 + cos_psi) / 2.0)"
    cos_omega = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    expr = "(cos(theta_view) + cos(theta_sun)) / (2.0 * cos_omega)"
    cos_beta = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    expr = "0.003 + 0.00512 * ws_"
    sigma2 = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    expr = "((1.0 / cos_beta**2) - 1.0) / sigma2"
    fac = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    expr = "exp(-fac) / (sigma2 * pi_)"
    pval = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    # tolerance mask
    expr = "fac > -log(1.0e-5)"
    tolerance_mask = numexpr.evaluate(expr)

    # insert 0.0 for any pixels that are flagged by the tolerance test
    pval[tolerance_mask] = 0.0

    expr = "pval * cos_omega / (4.0 * cos_beta**3)"
    gval = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    # value used for pixels that are flagged by the tolerance test
    value = numpy.abs((ri_ - 1) / (ri_ + 1)) ** 2

    expr = "arccos(cos_omega)"
    omega = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    expr = "arcsin(sin(omega) / ri_)"
    omega_prime = numexpr.evaluate(expr)  # noqa # pylint: disable=unused-variable

    # tolerance mask
    expr = "abs(omega + omega_prime) < 1.0e-5"
    tolerance_mask = numexpr.evaluate(expr)

    expr = (
        "p5_ * ((sin(omega-omega_prime) / sin(omega+omega_prime))**2 "
        "+ (tan(omega-omega_prime) / tan(omega+omega_prime))**2)"
    )

    pf_omega = numexpr.evaluate(expr)

    # insert value for any pixels that are flagged by the tolerance test
    pf_omega[tolerance_mask] = value

    expr = "pf_omega * gval * pi_"
    frsun = numexpr.evaluate(expr)

    return frsun


def sun_glint_correction(sky_glint_corrected, fs, sun_glint):
    """
    Apply sun glint correction.

    :param sky_glint_corrected:
       surface reflectance corrected for sky glint.

    :param fs:
        Direct fraction in the sun direction.

    :param sun_glint:
        sun glint.

    :return:
        A 2D numpy.ndarray of type float32.
    """
    expr = "sky_glint_corrected - fs * sun_glint"
    sun_glint_corrected = numexpr.evaluate(expr)

    return sun_glint_corrected
