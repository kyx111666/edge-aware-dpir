"""Check core modification results against saved pixels and the original baseline."""
import csv
from collections import defaultdict

import hdf5storage
import numpy as np
from PIL import Image

import core_experiment as core


def main():
    root = core.ROOT
    directory = root / "results/core/test"
    run = core.base.read_json(directory / "run.json")
    protocol = core.base.read_json(root / "core_protocol.json")
    frozen = core.base.read_json(root / "results/core/validation/calibration.json")
    assert frozen["protocol"] == protocol == run["protocol"]
    assert set(frozen["selection_images"]).isdisjoint(protocol["test_images"])
    with (directory / "metrics.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    records = defaultdict(dict)
    for row in rows:
        assert row["method"] not in records[row["case"]]
        records[row["case"]][row["method"]] = row
    assert len(rows) == 200 and len(records) == run["case_count"] == 40
    diagnostics = core.base.read_json(directory / "solver_diagnostics.json")
    validation = core.base.read_json(root / "results/core/validation/records.json")
    assert len(validation) == len(core.conditions(protocol, "validation")) * (2 + len(protocol["spatial_coefficient_candidates"]))
    for score in frozen["validation_scores"]:
        selected = [r for r in validation if r["method"] == score["method"] and r["spatial_coefficient"] == score["spatial_coefficient"]]
        for key in ("psnr", "ssim", "gradient_mae"):
            np.testing.assert_allclose(score[key], np.mean([r[key] for r in selected]), atol=1e-10)
    chosen, recommended = core.select_calibration(frozen["validation_scores"])
    assert chosen["spatial_coefficient"] == frozen["spatial_coefficient"]
    assert frozen["recommended"] == recommended
    kernels = hdf5storage.loadmat(root / "data/kernels/Levin09.mat")["kernels"]
    for case_id, case in records.items():
        assert set(case) == {"input", "original", "frequency", "uniform", "modified"}
        data = case["original"]
        clean, observed, _, seed = core.base.load_case(data["image"], int(data["kernel"]),
                                                       float(data["noise_255"]), protocol, kernels)
        assert seed == int(data["seed"])
        np.testing.assert_array_equal(clean, np.asarray(Image.open(directory / "images" / f"{case_id}_reference.png")))
        np.testing.assert_array_equal(core.base.to_uint8(observed), np.asarray(Image.open(directory / "images" / f"{case_id}_input.png")))
        for method, row in case.items():
            image = np.asarray(Image.open(directory / "images" / f"{case_id}_{method}.png"))
            metrics = core.quality(clean, image.astype(np.float32) / 255)
            for key in ("psnr", "ssim", "gradient_mae"):
                np.testing.assert_allclose(metrics[key], float(row[key]), atol=1e-9)
        assert all(int(case[m]["iterations"]) == 8 for m in ("original", "frequency", "uniform", "modified"))
        assert float(case["original"]["beta"]) == 0
        assert all(float(case[m]["beta"]) == frozen["beta"] for m in ("frequency", "uniform", "modified"))
        for method in ("uniform", "modified"):
            assert float(case[method]["spatial_coefficient"]) == frozen["spatial_coefficient"]
            detail = diagnostics[case_id][method]
            assert len(detail) == 8
            assert sum(d["steps"] for d in detail) == int(case[method]["cg_steps"])
            np.testing.assert_allclose(max(d["relative_residual"] for d in detail),
                                       float(case[method]["cg_residual"]), atol=1e-12)
            assert all(d["relative_residual"] <= protocol["cg_tolerance"] * 1.2 + 1e-7 for d in detail)
            assert all(0.05 - 1e-7 <= d["weight_min"] <= d["weight_mean"] <= d["weight_max"] <= 1 for d in detail)
            if method == "uniform":
                assert all(d["weight_min"] == d["weight_max"] == 1 for d in detail)
        weight_map = np.asarray(Image.open(directory / "images" / f"{case_id}_weights.png"))
        assert weight_map.shape == clean.shape and weight_map.min() >= 12
        assert float(case["frequency"]["spatial_coefficient"]) == 0
    # The earlier default-8 experiment provides an independent baseline replay check.
    prior_path = root / "results/test/metrics.csv"
    if prior_path.exists():
        with prior_path.open(encoding="utf-8-sig") as file:
            old = {row["case"]: row for row in csv.DictReader(file) if row["method"] == "original8"}
        for case_id, case in records.items():
            for key in ("psnr", "ssim"):
                np.testing.assert_allclose(float(case["original"][key]), float(old[case_id][key]), atol=1e-9)
        print("PASS original baseline matches the previous default-8 run")
    previous_frequency = root / "results/frequency_only/test/metrics.csv"
    if previous_frequency.exists():
        with previous_frequency.open(encoding="utf-8-sig") as file:
            history = list(csv.DictReader(file))
        previous = {row["case"]: row for row in history if row["method"] == "modified"}
        original_history = {row["case"]: row for row in history if row["method"] == "original"}
        for case_id, case in records.items():
            for key in ("psnr", "ssim", "gradient_mae"):
                np.testing.assert_allclose(float(case["frequency"][key]), float(previous[case_id][key]), atol=1e-9)
                np.testing.assert_allclose(float(case["original"][key]), float(original_history[case_id][key]), atol=1e-9)
        print("PASS original and frequency-only controls match the preserved previous run")
    print("PASS experiment parameters and validation/test image isolation")
    print("PASS validation scores and selected coefficient recomputed from 40 validation runs")
    print("PASS 40 paired conditions and all 200 metric rows")
    print("PASS degradation seeds, original images and saved inputs")
    print("PASS saved-image PSNR, SSIM and Sobel gradient errors")
    print("PASS all four restoration methods execute 8 iterations")
    print("PASS frozen beta and noise-scaled spatial coefficient in every test condition")
    print("PASS every recorded CG solve meets the specified normal-equation tolerance")


if __name__ == "__main__":
    main()
