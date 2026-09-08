import sys
from pathlib import Path

import auto_alignment.integration as integration

from mandible_registration.core_bridge import ensure_registration_core


def test_expected_public_integration_contract_is_installed():
    before = list(sys.path)
    package_dir = ensure_registration_core()
    assert sys.path == before
    assert integration.INTEGRATION_API_VERSION == 1
    assert integration.GENERAL_MODEL_REGISTRATION_VERSION == "2.0.0"
    assert package_dir == Path(integration.__path__[0]).resolve()


def test_required_public_names_are_available():
    from auto_alignment.integration import core, review, selection

    for name in ("AlignmentConfig", "MeshFacts", "RegistrationResult", "load_mesh", "register_meshes"):
        assert hasattr(core, name)
    for name in (
        "RegistrationReviewSpec",
        "build_review_manifest",
        "validate_review_manifest",
        "load_review_manifest",
        "run_general_result_viewer",
    ):
        assert hasattr(review, name)
    for name in (
        "SelectionRegionSpec",
        "RegionSelectionSession",
        "MultiRegionSelectionViewer",
        "encode_face_ranges",
        "decode_face_ranges",
    ):
        assert hasattr(selection, name)
