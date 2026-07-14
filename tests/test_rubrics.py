from smartedit.fusion.confidence import (
    cap_fallback,
    clamp_confidence,
    penalize_conflict,
)
from smartedit.fusion.rubrics import (
    score_background_music,
    score_catchy_music,
    score_length,
    score_narration,
    score_pace,
    score_semantic_signal,
    score_transitions,
)


def test_absent_narration_is_neutral_when_not_required() -> None:
    result = score_narration(
        20.0,
        {"speech_coverage": 0.0, "words_per_minute": 0.0},
        {},
        {"story": {"score": 1}, "text": {"score": 0}},
        {"personal": 0.8, "informational": 0.1, "promotional": 0.0},
        True,
    )
    assert result.score == 0


def test_missing_narration_is_negative_only_with_strong_need_evidence() -> None:
    result = score_narration(
        30.0,
        {"speech_coverage": 0.0, "words_per_minute": 0.0},
        {},
        {"story": {"score": -1}, "text": {"score": 0}},
        {"personal": 0.0, "informational": 0.9, "promotional": 0.0},
        True,
    )
    assert result.score == -1


def test_many_cuts_alone_do_not_make_pace_positive_or_negative() -> None:
    result = score_pace(
        30.0,
        {
            "shot_count": 31,
            "cuts_per_minute": 60.0,
            "average_shot_duration": 0.97,
        },
        {},
        {},
    )
    assert result.score == 0


def test_pace_conflict_lowers_confidence() -> None:
    result = score_pace(
        30.0,
        {
            "shot_count": 31,
            "cuts_per_minute": 60.0,
            "average_shot_duration": 0.97,
        },
        {"speech_coverage": 0.8, "words_per_minute": 240.0},
        {
            "pace": {
                "score": 1,
                "confidence": 0.9,
                "explanation": "Flow fits the demonstration.",
                "evidence": [],
            }
        },
    )
    assert result.score == 0
    assert result.confidence < 0.9
    assert result.conflicts


def test_no_background_music_is_not_negative() -> None:
    result = score_background_music(
        20.0,
        {"judgment": {"background_music_present": False}},
        True,
    )
    assert result.score == 0


def test_librosa_fallback_confidence_is_capped() -> None:
    result = score_background_music(
        20.0,
        {
            "fallback_used": True,
            "judgment": {
                "background_music_present": True,
                "background_music_score": 1,
                "confidence": 0.99,
            },
        },
        True,
    )
    assert result.confidence <= 0.45
    assert result.sources == ["librosa_fallback"]


def test_length_is_contextual_to_category() -> None:
    result = score_length(30.0, {"personal": 0.1, "informational": 0.8, "promotional": 0.2})
    assert result.score == 1


def test_transition_evidence_does_not_claim_unmeasured_gradual_candidates() -> None:
    result = score_transitions(
        20.0,
        {
            "shot_count": 3,
            "cut_count": 2,
            "cut_timestamps": [5.0, 12.0],
        },
        None,
    )

    observations = " ".join(item["observation"] for item in result.evidence).lower()
    assert "2 cuts" in observations
    assert "gradual" not in observations


def test_normal_speaking_rate_is_not_positive_without_semantic_support() -> None:
    result = score_narration(
        20.0,
        {"speech_coverage": 0.6, "speech_duration": 12.0, "words_per_minute": 140},
        {},
        {},
        {"informational": 0.9},
        True,
    )
    assert result.score == 0


def test_useful_well_timed_narration_can_be_positive() -> None:
    result = score_narration(
        20.0,
        {"speech_coverage": 0.6, "speech_duration": 12.0, "words_per_minute": 140},
        {
            "judgment": {
                "speech_music_interference": "low",
                "audio_quality": "good",
            }
        },
        {"story": {"score": 1}, "pace": {"score": 1}},
        {"informational": 0.9},
        True,
    )
    assert result.score == 1
    assert {"whisper", "audio_flamingo_3", "qwen3_vl"} <= set(result.sources)


def test_fast_narration_is_harmful() -> None:
    result = score_narration(
        20.0,
        {"speech_coverage": 0.7, "speech_duration": 14.0, "words_per_minute": 230},
        {},
        {},
        {},
        True,
    )
    assert result.score == -1


def test_audio_model_can_explicitly_flag_music_missing_where_needed() -> None:
    result = score_background_music(
        20.0,
        {
            "judgment": {
                "background_music_present": False,
                "background_music_score": -1,
                "evidence": [
                    {
                        "start": 0.0,
                        "end": 20.0,
                        "observation": "The long silent bed weakens the audible pacing.",
                    }
                ],
            }
        },
        True,
    )
    assert result.score == -1


def test_high_music_interference_is_harmful() -> None:
    result = score_background_music(
        15.0,
        {
            "judgment": {
                "background_music_present": True,
                "background_music_score": 1,
                "speech_music_interference": "high",
            }
        },
        True,
    )
    assert result.score == -1


def test_high_music_interference_is_harmful_to_narration() -> None:
    result = score_narration(
        15.0,
        {"speech_coverage": 0.8, "speech_duration": 12.0, "words_per_minute": 150},
        {
            "judgment": {
                "background_music_present": True,
                "speech_music_interference": "high",
                "audio_quality": "acceptable",
            }
        },
        {},
        {},
        True,
    )

    assert result.score == -1
    assert "audio_flamingo_3" in result.sources


def test_catchiness_absence_is_neutral_and_explicit_positive_survives() -> None:
    absent = score_catchy_music(15.0, {"judgment": {"background_music_present": False}}, True)
    positive = score_catchy_music(
        15.0,
        {
            "judgment": {
                "background_music_present": True,
                "catchy_music_score": 1,
                "catchiness_confidence": 0.85,
            }
        },
        True,
    )
    assert absent.score == 0
    assert positive.score == 1


def test_semantic_signal_preserves_valid_qwen_evidence() -> None:
    result = score_semantic_signal(
        "story",
        {
            "story": {
                "score": 1,
                "confidence": 0.8,
                "explanation": "The sequence resolves clearly.",
                "evidence": [{"start": 8.0, "end": 10.0, "observation": "The result is shown."}],
            }
        },
        10.0,
    )
    assert result.score == 1
    assert result.evidence[0]["end"] == 10.0


def test_transition_score_requires_explicit_transition_language() -> None:
    generic_effect = score_transitions(
        10.0,
        {"cut_count": 1},
        {
            "effects": {
                "score": 1,
                "confidence": 0.8,
                "explanation": "Color grading is consistent.",
                "evidence": [],
            }
        },
    )
    explicit_transition = score_transitions(
        10.0,
        {"cut_count": 1},
        {
            "effects": {
                "score": 1,
                "confidence": 0.8,
                "explanation": "The fade supports the scene change.",
                "evidence": [],
            }
        },
    )
    assert generic_effect.score == 0
    assert explicit_transition.score == 1


def test_length_extreme_and_unknown_category_are_conservative() -> None:
    extreme = score_length(300.0, {"informational": 0.9})
    unknown = score_length(30.0, {})
    assert extreme.score == -1
    assert unknown.score == 0


def test_confidence_helpers_are_deterministic_and_bounded() -> None:
    assert clamp_confidence(-1.0) == 0.0
    assert clamp_confidence(2.0) == 1.0
    assert penalize_conflict(0.8, 2) == 0.4
    assert cap_fallback(0.9, 0.4) == 0.4
