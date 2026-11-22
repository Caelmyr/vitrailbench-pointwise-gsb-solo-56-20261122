from types import SimpleNamespace

import numpy as np

from vitrail.bubbles import _census


def spot(x, y, area, size=1.0):
    return {"area_mm2": area, "centroid_mm": [x, y], "panel_position": [x/size, y/size],
            "panel_bounds": [(x-1)/size, (y-1)/size, (x+1)/size, (y+1)/size],
            "camera_position": [0, 0], "camera_pixel": [0, 0],
            "fitted_axes_mm": [1, 1], "angle_degrees": 0.0, "median_signal_difference": -.1,
            "near_unsupported_edge": False, "at_region_edge": False}


SCALE = SimpleNamespace(width_mm=100.0, height_mm=100.0)


def ctx(width_mm=100.0, height_mm=100.0):
    return SimpleNamespace(scale=SimpleNamespace(width_mm=width_mm, height_mm=height_mm))


def test_matching_bubbles_are_counted_once():
    first = [spot(10, 10, 4), spot(40, 40, 4)]
    second = [spot(10.2, 9.9, 4.1), spot(40.1, 40.2, 3.9)]
    entries, _ = _census(ctx(), first, second, 1.0)
    assert [entry["status"] for entry in entries] == ["confirmed", "confirmed"]
    assert sum(entry["count_low"] for entry in entries) == 2


def test_one_sided_spot_is_listed_separately():
    first = [spot(10, 10, 4)]
    second = [spot(10, 10, 4), spot(70, 70, 4)]
    entries, _ = _census(ctx(), first, second, 1.0)
    statuses = sorted(entry["status"] for entry in entries)
    assert statuses == ["confirmed", "second-only"]
    one_sided = next(entry for entry in entries if entry["status"] == "second-only")
    assert not one_sided["corroborated"]
    assert one_sided["count_low"] == one_sided["count_high"] == 1


def test_fused_blob_keeps_a_possible_range():
    first = [spot(30, 30, 4), spot(33, 30, 4)]
    second = [spot(31.5, 30, 8.2)]
    entries, _ = _census(ctx(), first, second, 1.0)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "merged"
    assert (entry["count_low"], entry["count_high"]) == (2, 2)
    assert entry["count_definite"]
    assert entry["area_sum_mm2"][0] == 8 and entry["area_sum_mm2"][1] == 8.2


def test_large_single_spot_can_hide_a_second():
    # The typical area comes from the neighbouring bubbles, so the large pair is compared against them.
    first = [spot(10, 10, 4), spot(60, 60, 4), spot(50, 50, 9)]
    second = [spot(10.1, 10.1, 4), spot(60.1, 60.1, 4), spot(50.3, 50.1, 9.1)]
    entries, _ = _census(ctx(), first, second, 1.0)
    large = max(entries, key=lambda entry: entry["area_sum_mm2"][0])
    assert large["status"] == "unresolved"
    assert (large["count_low"], large["count_high"]) == (1, 2)


def test_unequal_counts_without_fusion_keep_the_extra_spot():
    first = [spot(20, 20, 4)]
    second = [spot(20.1, 20.1, 4), spot(23, 20, 4)]
    entries, _ = _census(ctx(), first, second, 1.0)
    statuses = sorted(entry["status"] for entry in entries)
    assert statuses == ["confirmed", "second-only"]
    assert sum(entry["count_low"] for entry in entries) == 2


def test_pieces_scale_with_area():
    first = [spot(10, 10, 4), spot(50, 50, 4), spot(80, 80, 4)]
    second = [spot(10, 10, 12)]
    entries, _ = _census(ctx(), first, second, 1.0)
    fused = next(entry for entry in entries if entry["status"] != "confirmed")
    assert fused["count_high"] >= 2
