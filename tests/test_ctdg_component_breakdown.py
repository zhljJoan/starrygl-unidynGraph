import pytest

from atc_starrygl_lib.ctdg.train_loop import _component_breakdown


def test_component_breakdown_groups_sampling_communication_and_training() -> None:
    out = _component_breakdown(
        {
            "backend_sampling_seconds": 2.0,
            "backend_batch_build_seconds": 1.0,
            "backend_negative_attach_seconds": 0.5,
            "backend_submit_reads_seconds": 1.5,
            "backend_wait_patch_seconds": 0.5,
            "stage_memory_commit_submit_seconds": 0.25,
            "stage_memory_commit_wait_sync_seconds": 0.75,
            "stage_optimizer_sync_seconds": 0.1,
            "stage_optimizer_all_reduce_seconds": 0.2,
            "stage_encode_seconds": 3.0,
            "stage_head_loss_seconds": 1.0,
            "stage_backward_seconds": 2.0,
            "stage_optimizer_step_seconds": 0.6,
            "stage_memory_commit_build_seconds": 0.4,
            "stage_memory_commit_seconds": 0.3,
            "stage_wall_seconds": 15.0,
        }
    )

    assert out["component_sampling_seconds"] == pytest.approx(3.5)
    assert out["component_communication_seconds"] == pytest.approx(3.3)
    assert out["component_training_seconds"] == pytest.approx(7.3)
    assert out["component_unaccounted_seconds"] == pytest.approx(0.9)
