import numpy as np

from l4stack.localization.geodesy import GeodeticPosition, LocalTangentPlane


def test_local_tangent_plane_round_trip() -> None:
    origin = GeodeticPosition(40.1950, 29.0600, 120.0)
    plane = LocalTangentPlane(origin)
    expected = np.array([125.0, -48.0, 3.5])
    geodetic = plane.to_geodetic(*expected)
    recovered = plane.to_enu(geodetic)
    assert np.allclose(recovered, expected, atol=1e-4)
