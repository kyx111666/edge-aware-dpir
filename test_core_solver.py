"""Independent checks of the core reconstruction equation."""
import unittest

import numpy as np
import torch
from torch.nn import functional as functional

import core_experiment as core
from core_solver import (adaptive_data_solution, data_solution, edge_weights, gradient,
                         gradient_adjoint, laplacian_power, weighted_laplacian)


def circular_convolve(image, kernel):
    radius = kernel.shape[-1] // 2
    return functional.conv2d(functional.pad(image, (radius,) * 4, mode="circular"),
                             kernel.flip([-2, -1]))


def laplace(image):
    return 4 * image - sum(torch.roll(image, shift, axis)
                           for shift, axis in [(1, -2), (-1, -2), (1, -1), (-1, -1)])


class CountingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, tensor):
        self.calls += 1
        return tensor[:, :1]


class CoreSolverChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.rand(1, 1, 32, 32)
        self.y = torch.rand_like(self.x)
        self.kernel = torch.tensor([[0.01, 0.02, 0.03], [0.08, 0.3, 0.15], [0.2, 0.06, 0.15]])[None, None]
        self.spectral = core.base.sr.pre_calculate(self.y, self.kernel, 1)
        self.power = laplacian_power(self.x.shape, self.x)
        self.rho = torch.tensor(0.17).reshape(1, 1, 1, 1)

    def test_zero_weight_recovers_original_subproblem(self):
        actual = data_solution(self.x, self.spectral[2], self.spectral[3], self.rho, 0, self.power)
        original = core.base.sr.data_solution(self.x, *self.spectral, self.rho, 1)
        torch.testing.assert_close(actual, original, rtol=2e-6, atol=2e-6)

    def test_laplacian_response_matches_spatial_operator(self):
        lhs = torch.fft.fftn(laplace(laplace(self.x)), dim=(-2, -1))
        rhs = self.power * torch.fft.fftn(self.x, dim=(-2, -1))
        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-3)
        self.assertEqual(float(self.power[0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(self.power.max()), 64.0)

    def test_solution_satisfies_spatial_normal_equation(self):
        weight = 0.03
        result = data_solution(self.x, self.spectral[2], self.spectral[3], self.rho, weight, self.power)
        blurred = circular_convolve(result, self.kernel)
        lhs = circular_convolve(blurred + weight / 64 * laplace(laplace(blurred)), self.kernel.flip([-2, -1]))
        lhs = lhs + self.rho * result
        rhs = circular_convolve(self.y + weight / 64 * laplace(laplace(self.y)), self.kernel.flip([-2, -1])) + self.rho * self.x
        torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=2e-6)

    def test_high_frequency_data_fidelity_is_strengthened(self):
        checker = ((torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2).float()
        observed = (0.4 + 0.2 * checker)[None, None]
        spectral = core.base.sr.pre_calculate(observed, torch.ones(1, 1, 1, 1), 1)
        current = torch.full_like(observed, 0.5)
        original = data_solution(current, spectral[2], spectral[3], self.rho, 0, self.power)
        modified = data_solution(current, spectral[2], spectral[3], self.rho, 0.1, self.power)
        self.assertAlmostEqual(float(original.mean()), float(modified.mean()), places=6)
        self.assertGreater(float(modified.var()), float(original.var()))
        self.assertGreater(float(torch.linalg.vector_norm(original - modified)), 0.01)

    def test_data_consistent_image_is_not_smoothed(self):
        spectral = core.base.sr.pre_calculate(self.x, torch.ones(1, 1, 1, 1), 1)
        result = data_solution(self.x, spectral[2], spectral[3], self.rho, 3.0, self.power)
        torch.testing.assert_close(result, self.x, atol=3e-7, rtol=1e-6)

    def test_consistent_blurred_image_is_fixed_point(self):
        observed = circular_convolve(self.x, self.kernel)
        spectral = core.base.sr.pre_calculate(observed, self.kernel, 1)
        result = data_solution(self.x, spectral[2], spectral[3], self.rho, 3.0, self.power)
        torch.testing.assert_close(result, self.x, atol=3e-7, rtol=2e-6)

    def test_network_calls_are_eight_for_both_methods(self):
        protocol = core.base.read_json(core.ROOT / "core_protocol.json")
        observed = np.random.RandomState(2).rand(32, 32).astype(np.float32)
        for beta, coefficient, mode in [(0, 0, "adaptive"), (0.3, 0, "adaptive"),
                                         (0.3, 1, "uniform"), (0.3, 1, "adaptive")]:
            model = CountingDenoiser()
            restored, _ = core.restore(model, observed, np.ones((3, 3)) / 9, 7.65,
                                       protocol, beta, coefficient, mode)
            self.assertEqual(model.calls, 8)
            self.assertTrue(np.isfinite(restored).all())

    def test_gradient_adjoint_identity(self):
        horizontal, vertical = torch.randn_like(self.x), torch.randn_like(self.x)
        gx, gy = gradient(self.x)
        lhs = torch.sum(gx * horizontal + gy * vertical)
        rhs = torch.sum(self.x * gradient_adjoint(horizontal, vertical))
        torch.testing.assert_close(lhs, rhs)

    def test_weighted_operator_is_symmetric_positive(self):
        weights = edge_weights(self.x, 0.03)
        lhs = torch.sum(self.y * weighted_laplacian(self.x, weights))
        rhs = torch.sum(self.x * weighted_laplacian(self.y, weights))
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)
        self.assertGreaterEqual(float(torch.sum(self.x * weighted_laplacian(self.x, weights))), 0)

    def test_edges_get_less_smoothing_than_flat_regions(self):
        guide = torch.zeros_like(self.x)
        guide[..., 16:] = 1
        weights = edge_weights(guide, 0.01)
        self.assertEqual(float(weights[..., 5:10].min()), 1.0)
        self.assertLess(float(weights[..., 14:17].mean()), 0.2)
        self.assertGreaterEqual(float(weights.min()), 0.05 - 1e-7)

    def test_zero_spatial_strength_recovers_frequency_method(self):
        result, info = adaptive_data_solution(self.x, self.spectral[2], self.spectral[3],
            self.rho, 0.3, self.power, 0, torch.ones_like(self.x))
        expected = data_solution(self.x, self.spectral[2], self.spectral[3], self.rho, 0.3, self.power)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(info["steps"], 0)

    def test_uniform_weight_cg_matches_closed_form(self):
        strength = 0.01
        result, info = adaptive_data_solution(self.x, self.spectral[2], self.spectral[3],
            self.rho, 0.3, self.power, strength, torch.ones_like(self.x), 1e-6)
        q = 1 + 0.3 * self.power / 64
        expected = torch.fft.ifftn((q * self.spectral[3]
            + (self.rho + strength * self.power.sqrt()) * torch.fft.fftn(self.x, dim=(-2, -1)))
            / (q * self.spectral[2] + self.rho + strength * self.power.sqrt()), dim=(-2, -1)).real
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=1e-5)
        self.assertLess(info["relative_residual"], 1e-6)

    def test_adaptive_solution_matches_independent_dense_spatial_system(self):
        # Build H and D from spatial convolution/differences, independently of FFT/CG.
        size = 8
        basis = torch.eye(size * size, dtype=torch.float64).reshape(size * size, 1, size, size)
        kernel = self.kernel.double()
        blur = circular_convolve(basis, kernel).reshape(size * size, -1).T
        lap = laplace(basis).reshape(size * size, -1).T
        dx = (torch.roll(basis, -1, -1) - basis).reshape(size * size, -1).T
        dy = (torch.roll(basis, -1, -2) - basis).reshape(size * size, -1).T
        x, y = self.x[..., :size, :size].double(), self.y[..., :size, :size].double()
        weights = edge_weights(x, 0.03)
        wmatrix = torch.diag(weights.flatten())
        qmatrix = torch.eye(size * size, dtype=torch.float64) + 0.3 / 64 * lap.T @ lap
        rho, strength = 0.17, 0.05
        matrix = blur.T @ qmatrix @ blur + rho * torch.eye(size * size, dtype=torch.float64)
        matrix += strength * (dx.T @ wmatrix @ dx + dy.T @ wmatrix @ dy)
        rhs = blur.T @ qmatrix @ y.flatten() + rho * x.flatten()
        rhs += strength * (dx.T @ wmatrix @ dx + dy.T @ wmatrix @ dy) @ x.flatten()
        expected = torch.linalg.solve(matrix, rhs).reshape_as(x)
        spectral = core.base.sr.pre_calculate(y, kernel, 1)
        actual, info = adaptive_data_solution(x, spectral[2], spectral[3], torch.tensor(rho),
            0.3, laplacian_power(x.shape, x), strength, weights, 1e-9, 60)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
        self.assertLess(info["relative_residual"], 1e-9)

    def test_adaptive_solver_preserves_data_consistent_image(self):
        observed = circular_convolve(self.x, self.kernel)
        spectral = core.base.sr.pre_calculate(observed, self.kernel, 1)
        actual, info = adaptive_data_solution(self.x, spectral[2], spectral[3], self.rho,
            0.3, self.power, 0.2, edge_weights(self.x, 0.03), 1e-6)
        torch.testing.assert_close(actual, self.x, rtol=2e-6, atol=5e-7)
        self.assertLess(info["relative_residual"], 1e-6)

    def test_recommended_zero_is_not_presented_as_positive_modification(self):
        scores = [{"method": "frequency", "spatial_coefficient": 0, "psnr": 30.0},
                  {"method": "adaptive", "spatial_coefficient": 1, "psnr": 29.9},
                  {"method": "adaptive", "spatial_coefficient": 5, "psnr": 29.0}]
        study, recommended = core.select_calibration(scores)
        self.assertEqual(study["spatial_coefficient"], 1)
        self.assertEqual(recommended["spatial_coefficient"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
