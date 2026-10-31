"""Compare DPIR and frequency/spatial modifications at eight fixed iterations."""
import argparse
import csv
import itertools
import time

import hdf5storage
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import experiment as base
from core_solver import adaptive_data_solution, data_solution, edge_weights, laplacian_power

ROOT = base.ROOT


def quality(reference, estimate):
    metrics = base.quality(reference, estimate)
    restored = base.to_uint8(estimate).astype(np.float32)
    clean = reference.astype(np.float32)
    metrics["gradient_mae"] = float(sum(np.mean(np.abs(
        base.ndimage.sobel(restored, axis=axis, mode="wrap") -
        base.ndimage.sobel(clean, axis=axis, mode="wrap"))) / 8 for axis in (0, 1)) / 2)
    return metrics


@torch.inference_mode()
def restore(model, observed, kernel, noise_255, protocol, beta=0.0,
            spatial_coefficient=0.0, weight_mode="adaptive", diagnostics=None, weight_map_path=None):
    if noise_255 <= 0 or beta < 0 or spatial_coefficient < 0:
        raise ValueError("Noise must be positive; beta and spatial coefficient nonnegative")
    if weight_mode not in ("adaptive", "uniform"):
        raise ValueError("Unknown spatial weight mode")
    x = torch.from_numpy(observed.copy())[None, None]
    k = torch.from_numpy(kernel.astype(np.float32))[None, None]
    rhos, sigmas = base.pnp.get_rho_sigma(sigma=noise_255 / 255,
        iter_num=protocol["max_iterations"], modelSigma1=protocol["model_sigma_start_255"],
        modelSigma2=noise_255, w=1.0)
    start = time.perf_counter()
    spectral = base.sr.pre_calculate(x, k, 1)
    power = laplacian_power(x.shape, x) if beta or spatial_coefficient else None
    strength = spatial_coefficient * (noise_255 / 255)**2
    for index, (rho, sigma) in enumerate(zip(rhos, sigmas)):
        rho = torch.tensor(rho, dtype=x.dtype).reshape(1, 1, 1, 1)
        if spatial_coefficient:
            weights = edge_weights(x, noise_255 / 255) if weight_mode == "adaptive" else torch.ones_like(x)
            x, detail = adaptive_data_solution(x, spectral[2], spectral[3], rho, beta,
                power, strength, weights, protocol["cg_tolerance"], protocol["cg_max_steps"])
            if diagnostics is not None:
                diagnostics.append(dict(iteration=index + 1, **detail,
                    weight_min=float(weights.min()), weight_mean=float(weights.mean()),
                    weight_max=float(weights.max())))
        elif beta:
            x = data_solution(x, spectral[2], spectral[3], rho, beta, power)
        else:
            x = base.sr.data_solution(x, *spectral, rho, 1)
        x = base.denoise(model, x, sigma, index % 8 if protocol["x8"] else 0)
    elapsed = time.perf_counter() - start
    if not torch.isfinite(x).all():
        raise FloatingPointError("Non-finite reconstructed image")
    if weight_map_path is not None and spatial_coefficient:
        Image.fromarray(base.to_uint8(weights[0, 0].numpy())).save(weight_map_path)
    return x[0, 0].numpy(), elapsed


def conditions(protocol, split):
    return list(itertools.product(protocol[f"{split}_images"], protocol["kernel_indices"], protocol["noise_levels_255"]))


def select_calibration(scores):
    # Study a real positive-strength modification even if the best deployment choice is zero.
    study = max((r for r in scores if r["method"] == "adaptive" and r["spatial_coefficient"] > 0),
                key=lambda r: r["psnr"])
    recommended = max(scores, key=lambda r: r["psnr"])
    return study, recommended


def calibrate(model, protocol, kernels):
    rows = []
    beta = protocol["frequency_beta"]
    candidates = [("original", 0.0, 0.0), ("frequency", beta, 0.0)]
    candidates += [("adaptive", beta, value) for value in protocol["spatial_coefficient_candidates"]]
    products = conditions(protocol, "validation")
    for index, (name, kernel_index, noise) in enumerate(products):
        clean, observed, kernel, seed = base.load_case(name, kernel_index, noise, protocol, kernels)
        for method, frequency_beta, coefficient in candidates:
            restored, seconds = restore(model, observed, kernel, noise, protocol, frequency_beta, coefficient)
            rows.append(dict(image=name, kernel=kernel_index, noise_255=noise, method=method,
                             beta=frequency_beta, spatial_coefficient=coefficient,
                             seconds=seconds, seed=seed, **quality(clean, restored)))
        print(f"Core validation {index+1}/{len(products)} image={name} kernel={kernel_index} noise={noise}", flush=True)
    scores = []
    for method, frequency_beta, coefficient in candidates:
        selected = [r for r in rows if r["method"] == method and r["spatial_coefficient"] == coefficient]
        scores.append(dict(method=method, beta=frequency_beta, spatial_coefficient=coefficient,
            **{key: float(np.mean([r[key] for r in selected])) for key in ("psnr", "ssim", "gradient_mae")}))
    chosen, recommended = select_calibration(scores)
    frozen = {"protocol": protocol, "beta": chosen["beta"],
              "spatial_coefficient": chosen["spatial_coefficient"],
              "recommended": recommended,
              "validation_scores": scores, "selection_images": protocol["validation_images"],
              "frequency_beta_source": "Previous disjoint validation study; see results/frequency_only/validation/calibration.json",
              "selection_rule": "Study coefficient: best prespecified positive spatial value. Recommended configuration: best validation mean PSNR across all candidates, including spatial zero and original DPIR. Report both separately."}
    output = ROOT / "results/core/validation"
    base.write_json(output / "records.json", rows)
    base.write_json(output / "calibration.json", frozen)
    print("Core validation scores:", scores, "selected", chosen, flush=True)
    return frozen


def compare_image(path, reference, observed, estimates, scores):
    size = reference.shape[0]
    canvas = Image.new("RGB", (size * 3, (size + 50) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    panels = [("Reference", reference, None), ("Input", base.to_uint8(observed), "input"),
              ("Original DPIR 8", base.to_uint8(estimates["original"]), "original"),
              ("Frequency only", base.to_uint8(estimates["frequency"]), "frequency"),
              ("Uniform smoothing", base.to_uint8(estimates["uniform"]), "uniform"),
              ("Edge-aware modified", base.to_uint8(estimates["modified"]), "modified")]
    for index, (label, image, key) in enumerate(panels):
        left, top = index % 3 * size, index // 3 * (size + 50)
        canvas.paste(Image.fromarray(image).convert("RGB"), (left, top + 50))
        draw.text((left + 7, top + 3), label, fill="black", font=font)
        if key:
            draw.text((left + 7, top + 25), f"PSNR {scores[key]['psnr']:.2f}  SSIM {scores[key]['ssim']:.3f}", fill="black", font=font)
    canvas.save(path)


def evaluate(model, protocol, kernels, frozen, demo=False):
    if frozen["protocol"] != protocol:
        raise ValueError("Protocol changed; recalibrate before testing")
    output = ROOT / ("results/core/demo" if demo else "results/core/test")
    (output / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    all_diagnostics = {}
    products = conditions(protocol, "test")
    if demo:
        products = [products[0]]
    for index, (name, kernel_index, noise) in enumerate(products):
        clean, observed, kernel, seed = base.load_case(name, kernel_index, noise, protocol, kernels)
        case_id = f"img{name}_k{kernel_index}_n{noise:g}"
        scores = {"input": quality(clean, observed)}
        estimates = {}
        rows.append(dict(case=case_id, image=name, kernel=kernel_index, noise_255=noise, seed=seed,
                         method="input", beta=0, spatial_coefficient=0, iterations=0, seconds=0,
                         cg_steps=0, cg_residual=0, **scores["input"]))
        methods = ["original", "frequency", "uniform", "modified"]
        shift = index % len(methods)
        order = methods[shift:] + methods[:shift]
        all_diagnostics[case_id] = {}
        for method in order:
            beta = 0.0 if method == "original" else frozen["beta"]
            coefficient = frozen["spatial_coefficient"] if method in ("uniform", "modified") else 0.0
            detail = []
            map_path = output / "images" / f"{case_id}_weights.png" if method == "modified" else None
            estimates[method], seconds = restore(model, observed, kernel, noise, protocol, beta,
                coefficient, "uniform" if method == "uniform" else "adaptive", detail, map_path)
            all_diagnostics[case_id][method] = detail
            scores[method] = quality(clean, estimates[method])
            rows.append(dict(case=case_id, image=name, kernel=kernel_index, noise_255=noise, seed=seed,
                             method=method, beta=beta, spatial_coefficient=coefficient,
                             iterations=protocol["max_iterations"], seconds=seconds,
                             cg_steps=sum(d["steps"] for d in detail),
                             cg_residual=max((d["relative_residual"] for d in detail), default=0.0),
                             **scores[method]))
            Image.fromarray(base.to_uint8(estimates[method])).save(output / "images" / f"{case_id}_{method}.png")
        Image.fromarray(clean).save(output / "images" / f"{case_id}_reference.png")
        Image.fromarray(base.to_uint8(observed)).save(output / "images" / f"{case_id}_input.png")
        compare_image(output / "images" / f"{case_id}_comparison.png", clean, observed,
                       estimates, scores)
        print(f"Core test {index+1}/{len(products)} {case_id}: "
              f"original {scores['original']['psnr']:.3f}, frequency {scores['frequency']['psnr']:.3f}, "
              f"uniform {scores['uniform']['psnr']:.3f}, modified {scores['modified']['psnr']:.3f}", flush=True)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base.write_json(output / "solver_diagnostics.json", all_diagnostics)
    base.write_json(output / "run.json", {"protocol": protocol, "calibration": frozen,
                    "case_count": len(products), "core_change": "Frequency-weighted data fidelity plus edge-aware spatial gradient regularisation of z-x, solved by preconditioned CG",
                    "python": base.sys.version, "platform": base.platform.platform(),
                    "torch": torch.__version__, "numpy": np.__version__, "scipy": base.scipy.__version__,
                    "timing": "CPU wall time including spectral precompute, spatial weights, CG and all 8 network calls; excludes model loading, warmup, metrics and file I/O"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "validate", "test", "demo"], default="all")
    args = parser.parse_args()
    protocol = base.read_json(ROOT / "core_protocol.json")
    if set(protocol["validation_images"]) & set(protocol["test_images"]):
        raise ValueError("Validation and test images overlap")
    from prepare import prepare_assets
    prepare_assets()
    model = base.load_model(protocol)
    kernels = hdf5storage.loadmat(ROOT / "data/kernels/Levin09.mat")["kernels"]
    frozen = calibrate(model, protocol, kernels) if args.stage in ("all", "validate") else base.read_json(ROOT / "results/core/validation/calibration.json")
    if args.stage in ("all", "test", "demo"):
        evaluate(model, protocol, kernels, frozen, demo=args.stage == "demo")


if __name__ == "__main__":
    main()
