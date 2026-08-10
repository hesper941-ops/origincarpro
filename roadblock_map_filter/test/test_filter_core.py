import math

from roadblock_map_filter.filter_core import (
    CONFIRMED,
    TENTATIVE,
    Measurement,
    RoadblockMapFilterCore,
)


def measurement(x, y=0.0, raw_id=1):
    return Measurement(raw_id, x, y)


def stamp(frame):
    return (100, frame)


def confirm_one(core=None, x=1.0, y=0.5):
    core = core or RoadblockMapFilterCore()
    core.process_frame([measurement(x, y, 10)], stamp(1))
    core.process_frame([measurement(x + 0.02, y, 20)], stamp(2))
    core.process_frame([measurement(x + 0.01, y, 30)], stamp(3))
    assert len(core.confirmed_tracks()) == 1
    return core


def test_first_measurement_is_tentative_and_not_output():
    core = RoadblockMapFilterCore()
    result = core.process_frame([measurement(1.0)], stamp(1))
    assert [event.decision for event in result.events] == ['NEW_TENTATIVE']
    assert core.tracks[0].state == TENTATIVE
    assert core.confirmed_tracks() == ()


def test_third_close_measurement_confirms_track():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    second = core.process_frame([measurement(1.06)], stamp(2))
    result = core.process_frame([measurement(1.03)], stamp(3))
    assert 'CONFIRM' not in [event.decision for event in second.events]
    assert 'CONFIRM' in [event.decision for event in result.events]
    assert core.confirmed_tracks()[0].state == CONFIRMED


def test_tentative_expires_after_one_hit_in_five_frames():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    core.process_frame([], stamp(2))
    core.process_frame([], stamp(3))
    core.process_frame([], stamp(4))
    result = core.process_frame([], stamp(5))
    assert 'TENTATIVE_EXPIRE' in [event.decision for event in result.events]
    assert core.tracks == []


def test_confirmed_accept_updates_coordinate_median():
    core = confirm_one()
    before = core.confirmed_tracks()[0].stable_x
    result = core.process_frame([measurement(1.06, 0.5)], stamp(4))
    track = core.confirmed_tracks()[0]
    assert 'ACCEPT' in [event.decision for event in result.events]
    assert track.stable_x != before
    assert math.isclose(track.stable_x, 1.015, abs_tol=1e-9)


def test_confirmed_suspect_does_not_move_stable_position():
    core = confirm_one()
    track = core.confirmed_tracks()[0]
    before = (track.stable_x, track.stable_y, list(track.history))
    result = core.process_frame(
        [measurement(track.stable_x + 0.10, track.stable_y)], stamp(4)
    )
    track = core.confirmed_tracks()[0]
    assert 'SUSPECT' in [event.decision for event in result.events]
    assert (track.stable_x, track.stable_y, track.history) == before


def test_confirmed_track_persists_through_one_empty_frame():
    core = confirm_one()
    core.process_frame([], stamp(4))
    assert len(core.confirmed_tracks()) == 1


def test_confirmed_track_persists_through_many_empty_frames():
    core = confirm_one()
    track_id = core.confirmed_tracks()[0].track_id
    for frame in range(4, 104):
        core.process_frame([], stamp(frame))
    assert core.confirmed_tracks()[0].track_id == track_id


def test_greedy_association_is_one_to_one():
    core = RoadblockMapFilterCore()
    core.process_frame(
        [measurement(0.0, raw_id=1), measurement(0.20, raw_id=2)], stamp(1)
    )
    core.process_frame(
        [measurement(0.01, raw_id=3), measurement(0.21, raw_id=4)], stamp(2)
    )
    core.process_frame(
        [measurement(0.02, raw_id=7), measurement(0.22, raw_id=8)], stamp(3)
    )
    assert len(core.confirmed_tracks()) == 2
    result = core.process_frame(
        [measurement(0.09, raw_id=5), measurement(0.11, raw_id=6)], stamp(4)
    )
    matched_ids = [
        event.matched_track_id for event in result.events
        if event.measurement is not None
    ]
    assert len(matched_ids) == len(set(matched_ids)) == 2


def test_raw_id_order_swap_does_not_swap_track_identity():
    core = RoadblockMapFilterCore()
    core.process_frame(
        [measurement(0.0, raw_id=10), measurement(1.0, raw_id=20)], stamp(1)
    )
    core.process_frame(
        [measurement(1.01, raw_id=10), measurement(0.01, raw_id=20)], stamp(2)
    )
    core.process_frame(
        [measurement(0.02, raw_id=10), measurement(1.02, raw_id=20)], stamp(3)
    )
    tracks = core.confirmed_tracks()
    assert [track.track_id for track in tracks] == [1, 2]
    assert tracks[0].stable_x < tracks[1].stable_x


def test_single_outlier_does_not_create_second_confirmed_track():
    core = confirm_one()
    core.process_frame([measurement(1.40, 0.5, 99)], stamp(4))
    core.process_frame([], stamp(5))
    core.process_frame([], stamp(6))
    assert len(core.confirmed_tracks()) == 1


def test_five_point_coordinate_median():
    core = RoadblockMapFilterCore(history_size=5)
    points = [(1.00, 0.50), (1.02, 0.54), (0.98, 0.48),
              (1.06, 0.52), (1.01, 0.46)]
    for frame, (x, y) in enumerate(points, start=1):
        core.process_frame([measurement(x, y)], stamp(frame))
    track = core.confirmed_tracks()[0]
    assert len(track.history) == 5
    assert math.isclose(track.stable_x, 1.01, abs_tol=1e-9)
    assert math.isclose(track.stable_y, 0.50, abs_tol=1e-9)


def test_duplicate_timestamp_does_not_repeat_hit_or_history():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    duplicate = core.process_frame([measurement(1.01)], stamp(1))
    track = core.tracks[0]
    assert duplicate.duplicate_frame
    assert track.tentative_hits == 1
    assert track.tentative_age_frames == 1
    assert len(track.history) == 1
    core.process_frame([measurement(1.01)], stamp(2))
    assert core.tracks[0].tentative_hits == 2
    core.process_frame([measurement(1.02)], stamp(3))
    assert core.confirmed_tracks()[0].tentative_hits == 3


def test_zero_timestamp_fallback_processes_each_callback():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], None)
    core.process_frame([measurement(1.01)], None)
    core.process_frame([measurement(1.02)], None)
    assert len(core.confirmed_tracks()) == 1


def test_reset_clears_tracks_ids_and_timestamp_state():
    core = confirm_one()
    assert core.reset() == 1
    assert core.tracks == []
    assert core.next_track_id == 1
    assert core.last_processed_stamp is None
    core.process_frame([measurement(2.0)], stamp(1))
    assert core.tracks[0].track_id == 1


def test_nan_and_inf_are_skipped_safely():
    core = RoadblockMapFilterCore()
    result = core.process_frame([
        measurement(float('nan')),
        measurement(float('inf')),
        measurement(1.0),
    ], stamp(1))
    assert len(core.tracks) == 1
    assert sum(
        event.decision == 'INVALID_MEASUREMENT' for event in result.events
    ) == 2


def test_tentative_distance_between_confirm_and_association_gate_is_not_new():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    core.process_frame([measurement(1.10)], stamp(2))
    assert len(core.tracks) == 1
    assert core.tracks[0].tentative_hits == 1


def test_slow_drift_accept_then_suspect_does_not_relocate_or_duplicate():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.00, 0.50)], stamp(1))
    core.process_frame([measurement(1.00, 0.50)], stamp(2))
    core.process_frame([measurement(1.00, 0.50)], stamp(3))
    first = core.process_frame([measurement(1.06, 0.50)], stamp(4))
    stable_after_accept = core.confirmed_tracks()[0].stable_x
    second = core.process_frame([measurement(1.08, 0.50)], stamp(5))
    third = core.process_frame([measurement(1.10, 0.50)], stamp(6))
    assert 'ACCEPT' in [event.decision for event in first.events]
    assert 'SUSPECT' in [event.decision for event in second.events]
    assert 'SUSPECT' in [event.decision for event in third.events]
    assert core.confirmed_tracks()[0].stable_x == stable_after_accept
    assert len(core.confirmed_tracks()) == 1
    assert len(core.tracks) == 1


def test_confirmed_output_is_sorted_by_stable_track_id():
    core = RoadblockMapFilterCore()
    core.process_frame(
        [measurement(2.0), measurement(1.0), measurement(3.0)], stamp(1)
    )
    core.process_frame(
        [measurement(3.0), measurement(2.0), measurement(1.0)], stamp(2)
    )
    core.process_frame(
        [measurement(1.0), measurement(3.0), measurement(2.0)], stamp(3)
    )
    assert [track.track_id for track in core.confirmed_tracks()] == [1, 2, 3]


def test_two_hits_in_five_frames_do_not_confirm_and_expire():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    core.process_frame([measurement(1.01)], stamp(2))
    core.process_frame([], stamp(3))
    core.process_frame([], stamp(4))
    result = core.process_frame([], stamp(5))
    assert core.confirmed_tracks() == ()
    assert 'TENTATIVE_EXPIRE' in [event.decision for event in result.events]


def test_three_sequential_hits_confirm_on_third_frame():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    core.process_frame([measurement(1.01)], stamp(2))
    result = core.process_frame([measurement(1.02)], stamp(3))
    assert 'CONFIRM' in [event.decision for event in result.events]


def test_hits_on_frames_one_three_five_confirm():
    core = RoadblockMapFilterCore()
    core.process_frame([measurement(1.0)], stamp(1))
    core.process_frame([], stamp(2))
    core.process_frame([measurement(1.01)], stamp(3))
    core.process_frame([], stamp(4))
    result = core.process_frame([measurement(1.02)], stamp(5))
    assert 'CONFIRM' in [event.decision for event in result.events]


def _confirmed_at_origin():
    core = RoadblockMapFilterCore()
    for frame in range(1, 4):
        core.process_frame([measurement(0.0)], stamp(frame))
    return core


def test_new_track_suppression_boundary_is_inclusive():
    for offset in (0.179, 0.180):
        core = _confirmed_at_origin()
        result = core.process_frame([
            measurement(0.0, raw_id=2),
            measurement(offset, raw_id=3),
        ], stamp(4))
        assert 'SUPPRESS_NEAR_CONFIRMED' in [
            event.decision for event in result.events
        ]
        assert len(core.tracks) == 1


def test_new_track_just_outside_suppression_gate_is_tentative():
    core = _confirmed_at_origin()
    result = core.process_frame([
        measurement(0.0, raw_id=2),
        measurement(0.181, raw_id=3),
    ], stamp(4))
    assert 'NEW_TENTATIVE' in [event.decision for event in result.events]
    assert len(core.tracks) == 2


def test_far_new_obstacle_can_confirm_after_three_observations():
    core = RoadblockMapFilterCore()
    for frame in range(1, 4):
        core.process_frame([measurement(1.0)], stamp(frame))
    core.process_frame([measurement(1.30, raw_id=2)], stamp(4))
    core.process_frame([measurement(1.31, raw_id=2)], stamp(5))
    result = core.process_frame([measurement(1.29, raw_id=2)], stamp(6))
    assert 'CONFIRM' in [event.decision for event in result.events]
    assert len(core.confirmed_tracks()) == 2


def test_suppression_uses_nearest_of_all_confirmed_tracks():
    core = RoadblockMapFilterCore()
    for frame in range(1, 4):
        core.process_frame([
            measurement(0.0, raw_id=1),
            measurement(0.42, raw_id=2),
            measurement(1.0, raw_id=3),
        ], stamp(frame))
    result = core.process_frame([
        measurement(0.0, raw_id=4),
        measurement(0.42, raw_id=5),
        measurement(1.0, raw_id=6),
        measurement(0.25, raw_id=9),
    ], stamp(4))
    suppressed = next(
        event for event in result.events
        if event.decision == 'SUPPRESS_NEAR_CONFIRMED'
    )
    assert suppressed.matched_track_id == 2
    assert math.isclose(suppressed.distance_m, 0.17, abs_tol=1e-9)


def test_real_ghost_regression_suppresses_track_five_birth():
    core = RoadblockMapFilterCore()
    stable = (1.5295, 0.2432)
    for frame in range(1, 4):
        core.process_frame([measurement(*stable, raw_id=1)], stamp(frame))

    normal = measurement(1.5305, 0.2478, raw_id=8)
    ghost = measurement(1.3869, 0.1915, raw_id=9)
    result = core.process_frame([normal, ghost], stamp(4))
    decisions = {event.measurement.raw_id: event for event in result.events
                 if event.measurement is not None}
    assert decisions[8].decision == 'ACCEPT'
    assert decisions[9].decision == 'SUPPRESS_NEAR_CONFIRMED'
    assert decisions[9].matched_track_id == 1
    assert math.isclose(
        decisions[9].distance_m,
        math.hypot(ghost.x - stable[0], ghost.y - stable[1]),
        abs_tol=1e-9,
    )
    assert all(track.state == CONFIRMED for track in core.tracks)

    later = core.process_frame(
        [measurement(1.3957, 0.1953, raw_id=9)], stamp(5)
    )
    assert 'NEW_TENTATIVE' not in [event.decision for event in later.events]
    assert len(core.confirmed_tracks()) == 1
    assert len(core.tracks) == 1


def test_real_track_four_to_ten_case_is_reacquired_without_moving():
    core = RoadblockMapFilterCore()
    stable = (2.430, 2.087)
    initial = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), stable]
    for frame in range(1, 4):
        core.process_frame([
            measurement(x, y, raw_id=index)
            for index, (x, y) in enumerate(initial, start=1)
        ], stamp(frame))
    track = next(track for track in core.tracks if track.track_id == 4)
    history_before = list(track.history)

    new_position = measurement(2.452, 1.907, raw_id=10)
    result = core.process_frame([new_position], stamp(4))
    event = next(
        event for event in result.events
        if event.decision == 'REACQUIRE_SUSPECT'
    )
    assert event.matched_track_id == 4
    assert math.isclose(event.distance_m, math.hypot(0.022, 0.180))
    assert (event.stable_x_before, event.stable_y_before) == stable
    assert (event.stable_x_after, event.stable_y_after) == stable
    assert track.history == history_before
    assert not any(item.state == TENTATIVE for item in core.tracks)
    assert len(core.confirmed_tracks()) == 4


def test_real_track_one_to_twelve_case_is_reacquired():
    core = RoadblockMapFilterCore()
    stable = (1.766, 1.220)
    for frame in range(1, 4):
        core.process_frame([measurement(*stable)], stamp(frame))
    result = core.process_frame([
        measurement(1.848, 0.999, raw_id=12)
    ], stamp(4))
    event = next(
        event for event in result.events
        if event.decision == 'REACQUIRE_SUSPECT'
    )
    assert event.matched_track_id == 1
    assert math.isclose(event.distance_m, math.hypot(0.082, 0.221))
    assert (event.stable_x_before, event.stable_y_before) == stable
    assert (event.stable_x_after, event.stable_y_after) == stable
    assert len(core.tracks) == len(core.confirmed_tracks()) == 1


def test_reacquire_boundaries_and_first_pass_boundary():
    expected = {
        0.149: 'SUSPECT',
        0.150: 'SUSPECT',
        0.151: 'REACQUIRE_SUSPECT',
        0.249: 'REACQUIRE_SUSPECT',
        0.250: 'REACQUIRE_SUSPECT',
        0.251: 'NEW_TENTATIVE',
    }
    for distance, decision in expected.items():
        core = _confirmed_at_origin()
        result = core.process_frame([measurement(distance)], stamp(4))
        measurement_event = next(
            event for event in result.events if event.measurement is not None
        )
        assert measurement_event.decision == decision


def test_matched_confirmed_track_cannot_reacquire_22cm_measurement():
    core = _confirmed_at_origin()
    result = core.process_frame([
        measurement(0.01, raw_id=10),
        measurement(0.22, raw_id=20),
    ], stamp(4))
    events = {event.measurement.raw_id: event for event in result.events
              if event.measurement is not None}
    assert events[10].decision == 'ACCEPT'
    assert events[20].decision == 'NEW_TENTATIVE'
    assert all(event.decision != 'REACQUIRE_SUSPECT'
               for event in result.events)


def test_matched_confirmed_track_uses_v2_suppression_for_16cm_ghost():
    core = _confirmed_at_origin()
    result = core.process_frame([
        measurement(0.01, raw_id=10),
        measurement(0.16, raw_id=20),
    ], stamp(4))
    events = {event.measurement.raw_id: event for event in result.events
              if event.measurement is not None}
    assert events[10].decision == 'ACCEPT'
    assert events[20].decision == 'SUPPRESS_NEAR_CONFIRMED'
    assert len(core.tracks) == 1


def test_reacquire_second_pass_is_greedy_one_to_one():
    core = RoadblockMapFilterCore()
    track_one = (0.0, 0.0)
    track_two = (0.2897058823529411, 0.19639374157570258)
    for frame in range(1, 4):
        core.process_frame([
            measurement(*track_one, raw_id=1),
            measurement(*track_two, raw_id=2),
        ], stamp(frame))

    result = core.process_frame([
        measurement(0.17, 0.0, raw_id=11),
        measurement(0.1012144962300015, 0.17249819057864643, raw_id=12),
    ], stamp(4))
    reacquired = {
        event.measurement.raw_id: event.matched_track_id
        for event in result.events
        if event.decision == 'REACQUIRE_SUSPECT'
    }
    assert reacquired == {11: 1, 12: 2}


def test_reacquire_does_not_change_history_or_stable():
    core = RoadblockMapFilterCore()
    for frame in range(1, 4):
        core.process_frame([measurement(1.0, 1.0)], stamp(frame))
    track = core.confirmed_tracks()[0]
    before = (track.stable_x, track.stable_y, list(track.history))
    result = core.process_frame([measurement(1.20, 1.0)], stamp(4))
    event = next(event for event in result.events
                 if event.decision == 'REACQUIRE_SUSPECT')
    assert (track.stable_x, track.stable_y, track.history) == before
    assert event.track_state_before == CONFIRMED
    assert event.track_state_after == CONFIRMED
    assert event.stable_x_before == event.stable_x_after == 1.0
    assert event.stable_y_before == event.stable_y_after == 1.0


def test_reacquire_then_suspect_then_accept_keeps_identity():
    core = RoadblockMapFilterCore()
    for frame, x in enumerate((0.98, 1.00, 1.02), start=1):
        core.process_frame([measurement(x, 1.0)], stamp(frame))
    track = core.confirmed_tracks()[0]
    track_id = track.track_id
    history_before = list(track.history)

    reacquire = core.process_frame([measurement(1.20, 1.0)], stamp(4))
    assert track.history == history_before
    suspect = core.process_frame([measurement(1.14, 1.0)], stamp(5))
    assert track.history == history_before
    accept = core.process_frame([measurement(1.05, 1.0)], stamp(6))
    assert 'REACQUIRE_SUSPECT' in [event.decision for event in reacquire.events]
    assert track.history != history_before
    assert len(track.history) == len(history_before) + 1
    assert 'SUSPECT' in [event.decision for event in suspect.events]
    assert 'ACCEPT' in [event.decision for event in accept.events]
    assert core.confirmed_tracks()[0].track_id == track_id
    assert math.isclose(track.stable_x, 1.01, abs_tol=1e-9)


def test_reacquire_gate_must_cover_association_gate():
    try:
        RoadblockMapFilterCore(
            association_gate_m=0.15,
            reacquire_gate_m=0.149,
        )
    except ValueError as exc:
        assert 'reacquire_gate_m' in str(exc)
    else:
        raise AssertionError('invalid reacquire gate was accepted')


def _confirmed_with_tentative(tentative_x=0.22):
    core = _confirmed_at_origin()
    result = core.process_frame([
        measurement(0.0, raw_id=10),
        measurement(tentative_x, raw_id=11),
    ], stamp(4))
    assert 'NEW_TENTATIVE' in [event.decision for event in result.events]
    tentative = next(track for track in core.tracks if track.state == TENTATIVE)
    return core, tentative


def test_confirmed_reacquire_precedes_closer_tentative_association():
    core, tentative = _confirmed_with_tentative()
    result = core.process_frame([measurement(0.20, raw_id=20)], stamp(5))
    event = next(event for event in result.events
                 if event.measurement is not None)
    assert event.decision == 'REACQUIRE_SUSPECT'
    assert event.matched_track_id == 1
    assert math.isclose(event.distance_m, 0.20, abs_tol=1e-9)
    assert math.isclose(0.20 - tentative.stable_x, -0.02, abs_tol=1e-9)
    assert tentative.tentative_hits == 1


def test_tentative_can_match_when_confirmed_was_normally_matched():
    core, tentative = _confirmed_with_tentative()
    result = core.process_frame([
        measurement(0.01, raw_id=20),
        measurement(0.20, raw_id=21),
    ], stamp(5))
    events = {event.measurement.raw_id: event for event in result.events
              if event.measurement is not None}
    assert events[20].decision == 'ACCEPT'
    assert events[20].matched_track_id == 1
    assert events[21].decision == 'TENTATIVE_HIT'
    assert events[21].matched_track_id == tentative.track_id
    assert tentative.tentative_hits == 2


def test_tentative_matches_when_all_confirmed_are_beyond_reacquire_gate():
    core, tentative = _confirmed_with_tentative(tentative_x=0.30)
    result = core.process_frame([measurement(0.32, raw_id=20)], stamp(5))
    event = next(event for event in result.events
                 if event.measurement is not None)
    assert event.decision == 'TENTATIVE_HIT'
    assert event.matched_track_id == tentative.track_id
    assert tentative.tentative_hits == 2


def test_real_track_one_to_nine_priority_sequence_expires_candidate():
    core = RoadblockMapFilterCore()
    stable = (1.411, 0.504)
    for frame in range(1, 4):
        core.process_frame([measurement(*stable)], stamp(frame))

    first = core.process_frame([
        measurement(1.700, 0.504, raw_id=9)
    ], stamp(4))
    tentative = next(track for track in core.tracks if track.state == TENTATIVE)
    assert next(event for event in first.events
                if event.measurement is not None).decision == 'NEW_TENTATIVE'
    assert tentative.tentative_hits == 1

    sequence = (
        (5, 1.607, 0.196, 0.093),
        (6, 1.642, 0.231, 0.058),
        (7, 1.606, 0.195, 0.094),
    )
    for frame, x, old_distance, tentative_distance in sequence:
        result = core.process_frame([
            measurement(x, 0.504, raw_id=9)
        ], stamp(frame))
        event = next(event for event in result.events
                     if event.measurement is not None)
        assert event.decision == 'REACQUIRE_SUSPECT'
        assert event.matched_track_id == 1
        assert math.isclose(event.distance_m, old_distance, abs_tol=1e-9)
        assert math.isclose(
            abs(x - tentative.stable_x), tentative_distance, abs_tol=1e-9
        )
        assert tentative.tentative_hits == 1

    expired = core.process_frame([], stamp(8))
    assert 'TENTATIVE_EXPIRE' in [
        event.decision for event in expired.events
    ]
    assert len(core.confirmed_tracks()) == 1
    assert len(core.tracks) == 1
