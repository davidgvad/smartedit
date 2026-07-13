from smartedit.preprocessing.frame_sampler import build_sampling_plan


def test_sampling_plan_does_not_request_frame_past_decoder_count() -> None:
    plan = build_sampling_plan(
        duration_seconds=16.350,
        fps=29.970,
        max_frames=12,
        frame_count=490,
    )

    assert len(plan) == 12
    assert plan[-1][0] == 489


def test_sampling_plan_rounds_metadata_frame_count_instead_of_ceiling() -> None:
    plan = build_sampling_plan(
        duration_seconds=16.350,
        fps=29.970,
        max_frames=12,
    )

    assert plan[-1][0] == 489
