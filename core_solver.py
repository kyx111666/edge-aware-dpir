"""Frequency and edge-aware spatial modifications to DPIR's data subproblem.

For fixed denoiser input x and fixed edge weights W, solve
  min_z 0.5||Hz-y||^2 + beta/128*||L(Hz-y)||^2 + 0.5*rho||z-x||^2
        + 0.5*strength*sum(W*|D(z-x)|^2).
The frequency-only case has a closed form; spatial weights require a CG solve.
L is the standard four-neighbour Laplacian taught in the image-processing course.
"""
import torch
from torch.nn import functional as F


def laplacian_power(shape, reference):
    """Squared Fourier response of the discrete four-neighbour Laplacian."""
    fy = torch.fft.fftfreq(shape[-2], device=reference.device, dtype=reference.dtype)
    fx = torch.fft.fftfreq(shape[-1], device=reference.device, dtype=reference.dtype)
    response = 4 - 2 * torch.cos(2 * torch.pi * fy[:, None]) - 2 * torch.cos(2 * torch.pi * fx[None, :])
    return response.square()[None, None]


def data_solution(x, blur_power, observed_product, rho, weight, laplace_power):
    """Solve the modified non-blind deblurring subproblem in the frequency domain.

    observed_product = conj(H) * FFT(y); blur_power = abs(H)^2.
    Q = 1 + beta*|L|^2/64 weights the blur-data residual by frequency.
    This adds edge-domain data fidelity, while leaving the learned denoiser fixed.
    """
    if weight < 0 or torch.any(rho <= 0):
        raise ValueError("Regularisation weight must be nonnegative and rho positive")
    fidelity_weight = 1 + weight * laplace_power / 64
    numerator = fidelity_weight * observed_product + rho * torch.fft.fftn(x, dim=(-2, -1))
    denominator = fidelity_weight * blur_power + rho
    return torch.fft.ifftn(numerator / denominator, dim=(-2, -1)).real


def gradient(image):
    """Forward horizontal/vertical differences with the upstream circular boundary."""
    return (torch.roll(image, -1, -1) - image,
            torch.roll(image, -1, -2) - image)


def gradient_adjoint(horizontal, vertical):
    """Exact adjoint of gradient, needed for a positive semidefinite regularizer."""
    return (torch.roll(horizontal, 1, -1) - horizontal
            + torch.roll(vertical, 1, -2) - vertical)


def edge_weights(guide, noise_sigma):
    """Use a smoothed current estimate, never the clean reference, to protect edges."""
    if noise_sigma <= 0:
        raise ValueError("Noise standard deviation must be positive")
    # A separable 3x3 binomial approximation to a Gaussian suppresses noisy gradients.
    taps = guide.new_tensor([1, 2, 1]) / 4
    kernel = (taps[:, None] * taps[None, :])[None, None]
    smoothed = F.conv2d(F.pad(guide, (1, 1, 1, 1), mode="circular"), kernel)
    horizontal, vertical = gradient(smoothed)
    threshold = max(2 * noise_sigma, 0.01)
    return (threshold**2 / (threshold**2 + horizontal.square()
                           + vertical.square())).clamp_min(0.05)


def weighted_laplacian(image, weights):
    """D* W D: smoothing is weaker across locations marked as strong edges."""
    horizontal, vertical = gradient(image)
    return gradient_adjoint(weights * horizontal, weights * vertical)


def conjugate_gradient(operator, rhs, initial, precondition, tolerance=1e-5, max_steps=40):
    """Solve the positive definite spatially varying system without a dense matrix."""
    if tolerance <= 0 or max_steps < 1:
        raise ValueError("CG tolerance and maximum steps must be positive")
    solution = initial.clone()
    residual = rhs - operator(solution)
    scale = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    relative = float(torch.linalg.vector_norm(residual) / scale)
    if relative <= tolerance:
        return solution, {"steps": 0, "relative_residual": relative}
    preconditioned = precondition(residual)
    direction = preconditioned.clone()
    product = torch.sum(residual * preconditioned)
    for step in range(1, max_steps + 1):
        applied = operator(direction)
        curvature = torch.sum(direction * applied)
        if not torch.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError("CG encountered a nonpositive curvature")
        alpha = product / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        relative = float(torch.linalg.vector_norm(residual) / scale)
        if relative <= tolerance:
            break
        preconditioned = precondition(residual)
        next_product = torch.sum(residual * preconditioned)
        direction = preconditioned + (next_product / product) * direction
        product = next_product
    # Recompute the true residual rather than trusting only the CG recurrence.
    relative = float(torch.linalg.vector_norm(rhs - operator(solution)) / scale)
    if not torch.isfinite(solution).all() or relative > tolerance * 1.2 + 1e-7:
        raise FloatingPointError(f"CG did not meet tolerance: residual={relative:.3g}")
    return solution, {"steps": step, "relative_residual": relative}


def adaptive_data_solution(x, blur_power, observed_product, rho, beta,
                           laplace_power, strength, weights,
                           tolerance=1e-5, max_steps=40):
    """Add strength/2 * sum_p weights[p] * |D(z-x)[p]|^2 to the FFT objective.

    Weights are frozen within this subproblem and recomputed from the current
    estimate at the next outer iteration. No global convergence claim is made.
    """
    initial = data_solution(x, blur_power, observed_product, rho, beta, laplace_power)
    if strength < 0 or torch.any(weights <= 0) or weights.shape != x.shape:
        raise ValueError("Strength must be nonnegative and weights positive with image shape")
    if strength == 0:
        return initial, {"steps": 0, "relative_residual": 0.0}
    fidelity_weight = 1 + beta * laplace_power / 64
    diagonal = fidelity_weight * blur_power + rho
    rhs = torch.fft.ifftn(fidelity_weight * observed_product,
                         dim=(-2, -1)).real + rho * x
    rhs = rhs + strength * weighted_laplacian(x, weights)

    def operator(image):
        spectral = torch.fft.ifftn(diagonal * torch.fft.fftn(image, dim=(-2, -1)),
                                  dim=(-2, -1)).real
        return spectral + strength * weighted_laplacian(image, weights)

    # Constant-weight approximation is diagonal in Fourier space and accelerates CG.
    precondition_diagonal = diagonal + strength * weights.mean() * laplace_power.sqrt()

    def precondition(residual):
        return torch.fft.ifftn(torch.fft.fftn(residual, dim=(-2, -1))
                              / precondition_diagonal, dim=(-2, -1)).real

    return conjugate_gradient(operator, rhs, initial, precondition, tolerance, max_steps)
