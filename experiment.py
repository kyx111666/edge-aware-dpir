"""DPIR deblurring reproduction, validation-only calibration, and held-out evaluation.

Original FFT solver, schedule and DRUNet are imported unchanged from upstream.
The early stopping rule and the Wiener comparison are course-project additions.
"""
import argparse
import csv
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import hdf5storage
import numpy as np
import scipy
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream/DPIR"))
from models.network_unet import UNetRes
from utils import utils_pnp as pnp
from utils import utils_sisr as sr


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_uint8(image):
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def quality(reference, estimate):
    restored = to_uint8(estimate)
    return {
        "psnr": float(peak_signal_noise_ratio(reference, restored, data_range=255)),
        "ssim": float(structural_similarity(reference, restored, data_range=255,
                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
    }


def augment(image, mode):
    # Identical transforms to DPIR/utils/utils_image.py:augment_img_tensor4.
    if mode == 0:
        return image
    if mode == 1:
        return image.rot90(1, [2, 3]).flip([2])
    if mode == 2:
        return image.flip([2])
    if mode == 3:
        return image.rot90(3, [2, 3])
    if mode == 4:
        return image.rot90(2, [2, 3]).flip([2])
    if mode == 5:
        return image.rot90(1, [2, 3])
    if mode == 6:
        return image.rot90(2, [2, 3])
    if mode == 7:
        return image.rot90(3, [2, 3]).flip([2])
    raise ValueError("Augmentation mode must be 0..7")


def denoise(model, image, sigma, mode):
    transformed = augment(image, mode)
    height, width = transformed.shape[-2:]
    padded = torch.nn.functional.pad(transformed, (0, -width % 16, 0, -height % 16), mode="replicate")
    output = model(torch.cat((padded, torch.full_like(padded, float(sigma))), dim=1))
    output = output[..., :height, :width]
    inverse = 8 - mode if mode in (3, 5) else mode
    return augment(output, inverse)


def load_model(protocol):
    if protocol["device"] != "cpu":
        raise ValueError("This timing protocol supports CPU only")
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4,
                    act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose")
    weights = torch.load(ROOT / "weights/drunet_gray.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        model(torch.zeros(1, 2, protocol["image_size"], protocol["image_size"]))
    return model


def load_case(image_name, kernel_index, noise_255, protocol, kernels):
    reference = np.asarray(Image.open(ROOT / f"data/set12/{image_name}.png").convert("L"))
    size = protocol["image_size"]
    if min(reference.shape) < size:
        raise ValueError("Image smaller than configured crop size")
    top, left = ((dimension - size) // 2 for dimension in reference.shape)
    reference = reference[top:top + size, left:left + size].copy()
    kernel = kernels[0, kernel_index].astype(np.float64)
    # Preserve upstream uint8 blur quantisation and circular boundary conditions.
    observed = ndimage.convolve(reference, kernel, mode="wrap").astype(np.float32) / 255
    seed = protocol["seed"] + int(image_name) * 10000 + kernel_index * 1000 + round(noise_255 * 100)
    rng = np.random.RandomState(seed)
    observed += rng.normal(0, noise_255 / 255, observed.shape)
    return reference, observed, kernel, seed


def relative_change(current, previous):
    return float(torch.linalg.vector_norm(current - previous) /
                 torch.linalg.vector_norm(previous).clamp_min(1e-12))


def stopping_iteration(changes, tolerance, min_iterations, patience):
    stable = 0
    for iteration, change in enumerate(changes, start=1):
        stable = stable + 1 if iteration >= min_iterations and change <= tolerance else 0
        if stable >= patience:
            return iteration
    return len(changes)


@torch.inference_mode()
def solve_dpir(model, observed, kernel, noise_255, protocol, tolerance=None,
               reference=None):
    tensor = torch.from_numpy(observed.copy())[None, None]
    psf = torch.from_numpy(kernel.astype(np.float32))[None, None]
    rhos, sigmas = pnp.get_rho_sigma(
        sigma=max(0.255 / 255, noise_255 / 255), iter_num=protocol["max_iterations"],
        modelSigma1=protocol["model_sigma_start_255"], modelSigma2=noise_255, w=1.0)
    history, stable = [], 0
    elapsed = 0.0
    start = time.perf_counter()
    spectral = sr.pre_calculate(tensor, psf, 1)
    elapsed += time.perf_counter() - start
    for index, (rho, sigma) in enumerate(zip(rhos, sigmas)):
        start = time.perf_counter()
        previous = tensor
        tau = torch.tensor(rho, dtype=torch.float32).reshape(1, 1, 1, 1)
        tensor = sr.data_solution(tensor, *spectral, tau, 1)
        tensor = denoise(model, tensor, sigma, index % 8 if protocol["x8"] else 0)
        change = relative_change(tensor, previous)
        iteration = index + 1
        stable = stable + 1 if (tolerance is not None and iteration >= protocol["min_iterations"]
                               and change <= tolerance) else 0
        done = tolerance is not None and stable >= protocol["patience"]
        elapsed += time.perf_counter() - start
        record = {"iteration": iteration, "relative_change": change, "seconds": elapsed}
        # Reference metrics are for validation/analysis only and never enter the stopping rule.
        if reference is not None:
            record.update(quality(reference, tensor[0, 0].numpy()))
        history.append(record)
        if done:
            break
    output = tensor[0, 0].numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite DPIR output")
    return output, history, elapsed


def wiener_restore(observed, kernel, noise_255, factor):
    start = time.perf_counter()
    psf = torch.from_numpy(kernel.astype(np.float32))[None, None]
    transfer = sr.p2o(psf, observed.shape)[0, 0].numpy()
    balance = factor * (noise_255 / 255) ** 2 / max(float(observed.var()), 1e-6)
    spectrum = np.fft.fft2(observed)
    output = np.fft.ifft2(np.conj(transfer) * spectrum /
                         (np.abs(transfer) ** 2 + balance)).real.astype(np.float32)
    return output, time.perf_counter() - start


def calibrate(model, protocol, kernels):
    directory = ROOT / "results/validation"
    cases, wiener_scores = [], {factor: [] for factor in protocol["wiener_factor_candidates"]}
    products = list(itertools.product(protocol["validation_images"], protocol["kernel_indices"],
                                      protocol["noise_levels_255"]))
    for index, (name, kernel_index, noise) in enumerate(products):
        reference, observed, kernel, seed = load_case(name, kernel_index, noise, protocol, kernels)
        _, history, _ = solve_dpir(model, observed, kernel, noise, protocol, reference=reference)
        cases.append({"image": name, "kernel": kernel_index, "noise_255": noise,
                      "seed": seed, "history": history})
        for factor in wiener_scores:
            restored, _ = wiener_restore(observed, kernel, noise, factor)
            wiener_scores[factor].append(quality(reference, restored)["psnr"])
        print(f"Validation {index+1}/{len(products)} image={name} k={kernel_index} noise={noise}", flush=True)
    candidates = []
    for tolerance in protocol["tolerance_candidates"]:
        losses, iterations = [], []
        for case in cases:
            history = case["history"]
            stop = stopping_iteration([r["relative_change"] for r in history], tolerance,
                                      protocol["min_iterations"], protocol["patience"])
            iterations.append(stop)
            losses.append(history[-1]["psnr"] - history[stop - 1]["psnr"])
        candidates.append({"tolerance": tolerance, "mean_iterations": float(np.mean(iterations)),
                           "mean_psnr_loss": float(np.mean(losses)), "worst_psnr_loss": float(max(losses)),
                           "accepted": bool(np.mean(losses) <= protocol["validation_mean_psnr_loss_limit_db"]
                                            and max(losses) <= protocol["validation_worst_psnr_loss_limit_db"])})
    accepted = [r for r in candidates if r["accepted"]]
    tolerance = min(accepted, key=lambda r: (r["mean_iterations"], r["tolerance"]))["tolerance"] if accepted else 0.0
    factor = max(wiener_scores, key=lambda value: np.mean(wiener_scores[value]))
    frozen = {"protocol": protocol, "tolerance": tolerance,
              "wiener_factor": factor, "selection_images": protocol["validation_images"],
              "tolerance_candidates": candidates,
              "wiener_validation_psnr": {str(k): float(np.mean(v)) for k, v in wiener_scores.items()}}
    write_json(directory / "trajectories.json", cases)
    write_json(directory / "calibration.json", frozen)
    print("Frozen calibration:", tolerance, "Wiener factor:", factor, flush=True)
    return frozen


def comparison_image(path, reference, observed, estimates, scores):
    size = reference.shape[0]
    canvas = Image.new("RGB", (3 * size, 2 * (size + 52)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    panels = [("Reference", reference), ("Input", to_uint8(observed))]
    panels.extend((label, to_uint8(estimates[key])) for label, key in
                  [("Wiener", "wiener"), ("DPIR default 8", "original8"),
                   ("DPIR fixed 16", "fixed"), ("DPIR adaptive", "adaptive")])
    for index, (label, image) in enumerate(panels):
        left, top = (index % 3) * size, (index // 3) * (size + 52)
        canvas.paste(Image.fromarray(image).convert("RGB"), (left, top + 52))
        draw.text((left + 8, top + 5), label, fill="black", font=font)
        key = [None, "input", "wiener", "original8", "fixed", "adaptive"][index]
        if key:
            metric = scores[key]
            draw.text((left + 8, top + 27), f"PSNR {metric['psnr']:.2f}  SSIM {metric['ssim']:.3f}", fill="black", font=font)
    canvas.save(path)


def evaluate(model, protocol, kernels, frozen, demo=False):
    if frozen["protocol"] != protocol:
        raise ValueError("Protocol changed after calibration; run validation again")
    directory = ROOT / ("results/demo" if demo else "results/test")
    (directory / "images").mkdir(parents=True, exist_ok=True)
    names = protocol["test_images"][:1] if demo else protocol["test_images"]
    products = list(itertools.product(names, protocol["kernel_indices"], protocol["noise_levels_255"]))
    if demo:
        name = protocol["test_images"][min(1, len(protocol["test_images"]) - 1)]
        products = [(name, protocol["kernel_indices"][-1], protocol["noise_levels_255"][0])]
    rows, trajectories = [], []
    for index, (name, kernel_index, noise) in enumerate(products):
        reference, observed, kernel, seed = load_case(name, kernel_index, noise, protocol, kernels)
        case_id = f"img{name}_k{kernel_index}_n{noise:g}"
        scores = {"input": quality(reference, observed)}
        rows.append(dict(case=case_id, image=name, kernel=kernel_index, noise_255=noise, seed=seed,
                         method="input", iterations=0, seconds=0.0, **scores["input"]))
        estimates = {}
        estimates["wiener"], seconds = wiener_restore(observed, kernel, noise, frozen["wiener_factor"])
        scores["wiener"] = quality(reference, estimates["wiener"])
        rows.append(dict(case=case_id, image=name, kernel=kernel_index, noise_255=noise, seed=seed,
                         method="wiener", iterations=1, seconds=seconds, **scores["wiener"]))
        methods = ["original8", "fixed", "adaptive"]
        order = methods[index % 3:] + methods[:index % 3]
        for method in order:
            tolerance = frozen["tolerance"] if method == "adaptive" else None
            run_protocol = {**protocol, "max_iterations": 8} if method == "original8" else protocol
            estimates[method], history, seconds = solve_dpir(model, observed, kernel, noise, run_protocol,
                                                            tolerance=tolerance, reference=reference)
            scores[method] = quality(reference, estimates[method])
            rows.append(dict(case=case_id, image=name, kernel=kernel_index, noise_255=noise, seed=seed,
                             method=method, iterations=len(history), seconds=seconds, **scores[method]))
            trajectories.append({"case": case_id, "method": method, "history": history})
            Image.fromarray(to_uint8(estimates[method])).save(directory / "images" / f"{case_id}_{method}.png")
        Image.fromarray(reference).save(directory / "images" / f"{case_id}_reference.png")
        Image.fromarray(to_uint8(observed)).save(directory / "images" / f"{case_id}_input.png")
        Image.fromarray(to_uint8(estimates["wiener"])).save(directory / "images" / f"{case_id}_wiener.png")
        comparison_image(directory / "images" / f"{case_id}_comparison.png", reference, observed, estimates, scores)
        print(f"Test {index+1}/{len(products)} {case_id}: fixed {scores['fixed']['psnr']:.2f}, "
              f"adaptive {scores['adaptive']['psnr']:.2f}, "
              f"done", flush=True)
    with (directory / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "trajectories.json", trajectories)
    write_json(directory / "run.json", {"protocol": protocol, "calibration": frozen,
               "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
               "numpy": np.__version__, "scipy": scipy.__version__, "case_count": len(products),
               "timing": "CPU wall time: spectral precompute + iteration solve + stopping checks; excludes loading, metrics and file I/O"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "validate", "test", "demo"], default="all")
    args = parser.parse_args()
    protocol = read_json(ROOT / "protocol.json")
    if set(protocol["validation_images"]) & set(protocol["test_images"]):
        raise ValueError("Validation and test images overlap")
    from prepare import prepare_assets
    prepare_assets()
    model = load_model(protocol)
    kernels = hdf5storage.loadmat(ROOT / "data/kernels/Levin09.mat")["kernels"]
    if args.stage in ("all", "validate"):
        frozen = calibrate(model, protocol, kernels)
    elif args.stage == "demo":
        calibration = ROOT / "results/validation/calibration.json"
        frozen = read_json(calibration) if calibration.exists() else {
            "protocol": protocol, "tolerance": 0.005,
            "wiener_factor": 0.1, "demo_defaults": True, "selection_images": []}
    else:
        frozen = read_json(ROOT / "results/validation/calibration.json")
    if args.stage in ("all", "test", "demo"):
        evaluate(model, protocol, kernels, frozen, demo=args.stage == "demo")


if __name__ == "__main__":
    main()
