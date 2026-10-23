#!/usr/bin/env python3
"""Align bounded submission timestamp probes to CLOCK_MONOTONIC and frame flips.

TOP/BOTTOM interval unions include internal dependencies. They are not shader
time or GPU utilization. CPU recording intervals are wall spans, not CPU cycles.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics


def union(intervals, begin, end):
    result = []
    for a, b in sorted((max(a, begin), min(b, end)) for a, b in intervals
                       if b > begin and a < end):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b - a for a, b in intervals)


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def parse(path):
    records, flips, calibrations, summaries = [], [], defaultdict(list), []
    for line in path.read_text().splitlines():
        if not line.startswith("LOCAL_GPU_"):
            continue
        kind = line.split()[0]
        if kind not in {"LOCAL_GPU_CAL", "LOCAL_GPU_SUBMISSION", "LOCAL_GPU_FRAME", "LOCAL_GPU_SUMMARY"}:
            continue
        row = dict(re.findall(r"(\w+)=([^\s]+)", line))
        row = {k: (v if k == "trace" else float(v) if k == "period_ns" else int(v))
               for k, v in row.items()}
        if kind == "LOCAL_GPU_CAL": calibrations[row["trace"]].append(row)
        elif kind == "LOCAL_GPU_SUBMISSION": records.append(row)
        elif kind == "LOCAL_GPU_FRAME": flips.append(row)
        elif kind == "LOCAL_GPU_SUMMARY": summaries.append(row)
    if not records or not flips or not summaries:
        raise ValueError("Require completed real-game submission, calibration and flip records")
    if any(r["dropped"] or r["unavailable"] or r["records"] != r["started"] for r in summaries):
        raise ValueError("Trace contains dropped/unavailable timestamps")
    if sum(r["records"] for r in summaries) != len(records):
        raise ValueError("Submission record count differs from completion summary")
    for values in calibrations.values(): values.sort(key=lambda c: c["gpu"])
    # Both schedulers use the same VkDevice/queue. Share its calibration samples:
    # the presentation scheduler alone only calibrates every 1024 flips. Keeping
    # its offset fixed for seconds ignores drift between the two clocks.
    clock = sorted((c for values in calibrations.values() for c in values), key=lambda c: c["gpu"])
    if len(clock) < 2 or len({c["period_ns"] for c in clock}) != 1:
        raise ValueError("Require one device clock with multiple calibrations")
    ticks = [c["gpu"] for c in clock]
    def aligned(tick):
        index = bisect.bisect_left(ticks, tick)
        if index == 0 or index == len(clock):
            raise ValueError("Timestamp is outside bracketing calibrations")
        left, right = clock[index - 1:index + 1]
        if right["cpu_ns"] <= left["cpu_ns"]:
            raise ValueError("Calibration clocks are not monotonic")
        ratio = (tick - left["gpu"]) / (right["gpu"] - left["gpu"])
        value = round(left["cpu_ns"] + ratio * (right["cpu_ns"] - left["cpu_ns"]))
        # Report calibration endpoint uncertainty, not a claim that the clocks
        # cannot change rate between samples. The mapping is piecewise linear.
        return value, max(left["deviation_ns"], right["deviation_ns"]), right["cpu_ns"] - left["cpu_ns"]
    for r in records:
        r["gpu_begin_ns"], d1, g1 = aligned(r["gpu_start"])
        r["gpu_end_ns"], d2, g2 = aligned(r["gpu_end"])
        r["calibration_deviation_ns"] = max(d1, d2)
        r["calibration_bracket_ns"] = max(g1, g2)
        if r["gpu_end_ns"] < r["gpu_begin_ns"]:
            raise ValueError("GPU timestamp order reversed")
        if r["gpu_begin_ns"] + max(d1, d2) < r["cpu_end_record_ns"]:
            raise ValueError("GPU timestamp precedes recording completion beyond calibration uncertainty")
    return records, sorted(flips, key=lambda r: r["cpu_ns"]), calibrations, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--windows", type=Path, required=True, help="abba-switch.py report.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, flips, calibrations, summaries = parse(args.log)
    intervals = [(r["gpu_begin_ns"], r["gpu_end_ns"]) for r in records]
    ab = json.loads(args.windows.read_text())
    renderer_tid = Counter(r["tid"] for r in records).most_common(1)[0][0]
    windows, frame_rows = [], []
    for w in ab["windows"]:
        if w["value"] != 1: continue
        phase = json.loads((args.windows.parent / (w["label"] + ".json")).read_text())
        begin, end = phase["start_ns"], phase["end_ns"]
        batches = [r for r in records if begin <= r["cpu_submit_ns"] < end]
        merged = union(intervals, begin, end)
        gpu = duration(merged)
        latency = [(r["gpu_begin_ns"] - r["cpu_end_record_ns"]) / 1000 for r in batches]
        gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
        selected_flips = [f for f in flips if begin <= f["cpu_ns"] <= end]
        current_frames = []
        for a, b in zip(selected_flips, selected_flips[1:]):
            if b["frame"] != a["frame"] + 1: raise ValueError("Missing frame boundary inside capture")
            x, y = a["cpu_ns"], b["cpu_ns"]
            frame = {"window": w["index"], "frame": b["frame"], "begin_ns": x, "end_ns": y,
                     "frame_ms": (y - x) / 1e6,
                     "gpu_span_union_ms": duration(union(intervals, x, y)) / 1e6,
                     "submissions": sum(x <= r["cpu_submit_ns"] < y for r in batches)}
            current_frames.append(frame)
        frame_rows.extend(current_frames)
        windows.append({"index": w["index"], "begin_ns": begin, "end_ns": end,
                        "frames": phase["frames"], "fps": phase["measured_fps"],
                        "frame_ms": (end - begin) / phase["frames"] / 1e6,
                        "gpu_span_union_ms_per_counter_frame": gpu / phase["frames"] / 1e6,
                        "gpu_span_union_percent_of_window": gpu / (end - begin) * 100,
                        "outside_gpu_spans_ms_per_counter_frame": (end - begin - gpu) / phase["frames"] / 1e6,
                        "submissions": len(batches), "submissions_per_counter_frame": len(batches) / phase["frames"],
                        "record_end_to_gpu_start_us_median": percentile(latency, .5),
                        "record_end_to_gpu_start_us_p95": percentile(latency, .95),
                        "record_end_to_gpu_start_us_max": max(latency),
                        "internal_gpu_gap_count": len(gaps),
                        "internal_gpu_gaps_over_100us": sum(x > 100000 for x in gaps),
                        "internal_gpu_gap_ms_p95": percentile([g / 1e6 for g in gaps], .95),
                        "internal_gpu_gap_ms_max": max(gaps) / 1e6 if gaps else 0,
                        "complete_frame_intervals": len(current_frames)})
    if not frame_rows: raise ValueError("No complete measured frame intervals")
    offsets = {trace: [(c["cpu_ns"] - c["gpu"] * c["period_ns"]) for c in values]
               for trace, values in calibrations.items()}
    report = {"log": str(args.log.resolve()), "windows_report": str(args.windows.resolve()),
              "scope": "Submission TOP/BOTTOM timestamp unions, calibrated to CLOCK_MONOTONIC. "
                       "Includes dependencies; neither shader time nor GPU utilization. CPU record spans include waits/preemption.",
              "summary": summaries, "renderer_tid_by_submission_count": renderer_tid,
              "clock_alignment": {"method": "Shared VkDevice calibrations, bracketing piecewise linear mapping",
                                  "maximum_record_calibration_bracket_ms": max(r["calibration_bracket_ns"] for r in records) / 1e6,
                                  "maximum_endpoint_deviation_ns": max(r["calibration_deviation_ns"] for r in records),
                                  "limits": "Endpoint deviation is not a rigorous bound on interpolation drift between samples."},
              "calibrations": {k: {"count": len(v), "max_deviation_ns": max(c["deviation_ns"] for c in v),
                                     "offset_range_ns": max(offsets[k]) - min(offsets[k])}
                               for k, v in calibrations.items()},
              "windows": windows, "timing_off_on_comparison": ab["combined"], "frames": frame_rows}

    # Select the frame nearest the median, then show it and its neighbours.
    median = statistics.median(f["frame_ms"] for f in frame_rows)
    representative = min(frame_rows[1:-1], key=lambda f: abs(f["frame_ms"] - median))
    index = frame_rows.index(representative)
    selection = [f for f in frame_rows[max(0, index - 1):index + 2] if f["window"] == representative["window"]]
    a, b = selection[0]["begin_ns"], selection[-1]["end_ns"]
    report["representative"] = {"rule": "Frame nearest the median measured frame duration, with neighbours",
                                "median_frame_ms": median, "frame": representative["frame"],
                                "begin_ns": a, "end_ns": b}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 4.6))
    lanes = [("CPU recording (wall span)", 3, "#377eb8"),
             ("CPU finish + queue submit", 2, "#ff8c42"),
             ("GPU submission span union", 1, "#2ca67d")]
    spans = {3: [(r["cpu_record_ns"], r["cpu_end_record_ns"]) for r in records if r["tid"] == renderer_tid],
             2: [(r["cpu_end_record_ns"], r["cpu_submit_ns"]) for r in records if r["tid"] == renderer_tid],
             1: intervals}
    for name, lane, color in lanes:
        parts = union(spans[lane], a, b)
        ax.broken_barh([((x - a) / 1e6, (y - x) / 1e6) for x, y in parts], (lane - .23, .46), facecolors=color)
    for f in selection:
        ax.axvline((f["begin_ns"] - a) / 1e6, color="#777777", linewidth=.6)
        ax.text(((f["begin_ns"] + f["end_ns"]) / 2 - a) / 1e6, .42,
                f'Frame {f["frame"]}\n{f["frame_ms"]:.2f} ms; GPU spans {f["gpu_span_union_ms"]:.2f} ms',
                ha="center", va="center", fontsize=9)
    ax.set_yticks([3, 2, 1], [x[0] for x in lanes])
    ax.set_xlim(0, (b - a) / 1e6); ax.set_ylim(.05, 3.6)
    ax.set_xlabel("Milliseconds, aligned with calibrated CLOCK_MONOTONIC timestamps")
    ax.set_title("Demon's Souls W8 stationary scene — CPU / GPU submission timeline", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="x", alpha=.2)
    fig.text(.02, .015, "Diagnostic capture. GPU spans include dependencies; CPU recording spans include waits and preemption. No shader-time claim.", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(args.out / "timeline.png", dpi=170)
    fig.savefig(args.out / "timeline.svg")
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["summary", "calibrations", "windows", "representative"]}, indent=2))


if __name__ == "__main__":
    main()
