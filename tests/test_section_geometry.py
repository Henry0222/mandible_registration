import numpy as np
import pytest

from mandible_registration.section_geometry import (
    SectionState, sphere_roi, clip_segments_to_sphere, INITIAL_ORIENTATION,
)


def test_roi_includes_triangle_crossing_sphere_with_all_vertices_outside():
    vertices = np.array([[-100, -100, 0], [100, -100, 0], [0, 100, 0],
                         [21, 0, 0], [22, 1, 0], [22, -1, 0]], dtype=float)
    roi = sphere_roi(vertices, [[0, 1, 2], [3, 4, 5]], [0, 0, 0])
    assert roi.source_face_ids.tolist() == [0]
    assert len(roi.vertices) == 3
    assert roi.source_triangle_count == 2


def test_roi_rejects_diagonal_box_false_positive_and_keeps_tangency():
    vertices = np.array([[19, 19, 0], [19, -19, 0], [40, 0, 0],
                         [20, 0, 0], [21, 1, 0], [21, -1, 0],
                         [30, 0, 0], [0, 30, 0], [30, 30, 0]])
    roi = sphere_roi(vertices, [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [0, 0, 0])
    assert roi.source_face_ids.tolist() == [0, 1]


def test_roi_distance_is_translation_invariant_and_compact():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [200, 200, 200]])
    translation = np.array([500., -90., 200.])
    roi = sphere_roi(vertices + translation, [[0, 1, 2]], translation)
    assert len(roi.vertices) == 3
    np.testing.assert_allclose(roi.vertices, vertices[:3] + translation)


def test_empty_and_degenerate_roi_are_supported():
    assert sphere_roi(np.empty((0, 3)), np.empty((0, 3), int), [0, 0, 0]).triangles.shape == (0, 3)
    roi = sphere_roi(np.array([[20., 0, 0]]), [[0, 0, 0]], [0, 0, 0])
    assert roi.source_face_ids.tolist() == [0]


def test_exact_segment_clipping_never_exceeds_20_mm_or_fills_gaps():
    segments = [[[-30, 0, 0], [30, 0, 0]], [[-30, 21, 0], [30, 21, 0]],
                [[0, 0, 0], [0, 30, 0]], [[0, 1, 0], [0, 1, 0]]]
    clipped, ids = clip_segments_to_sphere(segments, [0, 0, 0])
    assert ids.tolist() == [0, 2]
    np.testing.assert_allclose(clipped, [[[-20, 0, 0], [20, 0, 0]], [[0, 0, 0], [0, 20, 0]]])
    assert np.max(np.linalg.norm(clipped, axis=2)) <= 20 + 1e-10


def test_initial_plane_world_yz_and_scroll_follows_rotated_camera():
    state = SectionState([4, 5, 6])
    assert "非解剖" in INITIAL_ORIENTATION
    state.scroll(1)
    np.testing.assert_allclose(state.origin, [3.5, 5, 6])
    state.rotate(225, 0)
    before = state.origin.copy()
    state.scroll(1, .1)
    np.testing.assert_allclose(state.origin - before, -.1 * state.normal, atol=1e-10)
    assert np.dot(state.up, state.normal) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(state.normal) == pytest.approx(1)
    state.scroll(1000)
    assert state.offset_mm == 20
    assert state.disk_radius == 0


def test_long_rotations_stay_orthonormal_and_plane_origin_keeps_offset():
    state = SectionState([0, 0, 0])
    state.scroll(10)
    for _ in range(200):
        state.rotate(13, -17)
    assert np.dot(state.normal, state.up) == pytest.approx(0, abs=1e-12)
    assert np.linalg.norm(state.right) == pytest.approx(1)
    assert np.dot(state.origin - state.center, state.normal) == pytest.approx(-5)


@pytest.mark.parametrize("center", [[0, 0], [0, 0, np.nan]])
def test_invalid_centers_rejected(center):
    with pytest.raises(ValueError):
        SectionState(center)
