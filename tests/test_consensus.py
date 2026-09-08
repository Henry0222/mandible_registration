from dataclasses import asdict, dataclass, replace

import numpy as np
import pytest

from mandible_registration import workflow
from auto_alignment.integration.core import RegistrationMetrics, RegistrationResult


@dataclass
class Quality:
    target_coverage_ratio: float = 0.5
    directed_overlap_ratio: float = 0.5
    normal_consistency_ratio: float = 0.9
    residual_p90_mm: float = 0.2

    def as_dict(self):
        return asdict(self)


def result(x, quality=None):
    transform = np.eye(4)
    transform[0, 3] = x
    return RegistrationResult(transform, "success", "高", RegistrationMetrics(1, .01, 100, 1, 0, abs(x)), (), .01, quality or Quality())


def test_fourth_attempt_recovers_without_lowering_quality_gate(tmp_path, monkeypatch):
    candidates = iter([result(100, Quality(target_coverage_ratio=.1)), result(0), result(10), result(.1, Quality(target_coverage_ratio=.7))])
    calls = []

    def register(*args):
        calls.append(args[4].random_seed)
        return next(candidates)

    monkeypatch.setattr(workflow, "register_meshes", register)
    chosen, diagnostics = workflow._register_ct_with_consensus(None, None, None, None, workflow.AlignmentConfig(), tmp_path, None)
    assert len(calls) == 4 and len(set(calls)) == 4
    assert diagnostics["consensus_members"] == [2, 4]
    assert diagnostics["selected_attempt"] == 4
    assert chosen.transformation[0, 3] == .1
    assert diagnostics["maximum_attempts"] == 5


@pytest.mark.parametrize("weak", (True, False))
def test_five_attempt_limit_rejects_weak_or_disagreeing_results(tmp_path, monkeypatch, weak):
    calls = []

    def register(*args):
        calls.append(args[4].random_seed)
        return result(0 if weak else len(calls) * 10, Quality(normal_consistency_ratio=.5) if weak else None)

    monkeypatch.setattr(workflow, "register_meshes", register)
    with pytest.raises(workflow.WorkflowError) as error:
        workflow._register_ct_with_consensus(None, None, None, None, workflow.AlignmentConfig(), tmp_path, None)
    assert len(calls) == 5
    assert error.value.candidate is not None
    assert all(not pair["agrees"] for pair in error.value.diagnostics["pairwise_consistency"])


@pytest.mark.parametrize("changes", ({"target_coverage_ratio": .24}, {"normal_consistency_ratio": .69}, {"residual_p90_mm": .81}, {"residual_p90_mm": None}))
def test_quality_thresholds_are_retained(changes):
    assert workflow._ct_attempt_is_strong(result(0))
    assert not workflow._ct_attempt_is_strong(result(0, replace(Quality(), **changes)))
