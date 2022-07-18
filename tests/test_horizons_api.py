import pytest
import numpy as np
import sys
import numpy.testing as npt
sys.path.insert(0, '../src')


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            ['Sun', 2455000],
            [1, 1.0, 4.6504672609621574e-03, [0., 0., 0.], [0., 0., 0.]],
            None,
        ),
        (
            ['Jupiter', 2455000],
            [1, 0.0009547919099366768, 4.778945025452157e-04,
            [3.68677697, -3.46246655, -0.06812788],
            [1.85424572,  2.14204681, -0.05038653]],
            None,
        ),
        (
            ['JuPiTer', 2455000],
            [1, 0.0009547919099366768, 4.778945025452157e-04,
            [3.68677697, -3.46246655, -0.06812788],
            [1.85424572, 2.14204681, -0.05038653]],
            None,
        ),
        (
            ['planet9', 2455000],
            [0, 0.,0., [0., 0., 0.], [0., 0., 0.]],
            KeyError,
        ),
    ])
def test_query_horizons_planets(test, expected, expect_raises):
    from horizons_api import query_horizons_planets
    xe = np.zeros(3)
    ve = np.zeros(3)
    xt = np.zeros(3)
    vt = np.zeros(3)
    fe, me, re, xe, ve = expected
    planet, epoch = test
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            ft, mt, rt, xt, vt = query_horizons_planets(planet, epoch)
            npt.assert_equal(int(ft), int(fe))
            npt.assert_almost_equal(xt, xe, decimal=3)
            npt.assert_almost_equal(vt, ve, decimal=3)
            npt.assert_almost_equal(me, mt, decimal=3)
            npt.assert_almost_equal(re, rt, decimal=3)
    else:
        ft, mt, rt, xt, vt = query_horizons_planets(planet, epoch)
        npt.assert_equal(int(ft), int(fe))
        npt.assert_almost_equal(xt, xe, decimal=3)
        npt.assert_almost_equal(vt, ve, decimal=3)
        npt.assert_almost_equal(me, mt, decimal=3)
        npt.assert_almost_equal(re, rt, decimal=3)
    return
