import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from src.stellar_stream.stream import StellarStream


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def simple_stream():
    phi1 = np.linspace(0, 10, 100)
    phi2 = np.sin(phi1)
    vlos = phi2 + np.random.normal(0, 0.1, size=phi2.shape)
    return StellarStream(phi1, phi2, vlos, name="SimpleStream")


@pytest.fixture
def constant_stream():
    phi1 = np.linspace(0, 10, 50)
    phi2 = np.zeros_like(phi1)
    vlos = np.ones_like(phi1) * 10
    return StellarStream(phi1, phi2, vlos, name="ConstantStream")


# ----------------------------
# Basic construction
# ----------------------------
def test_basic_init():
    phi1 = np.array([0, 1, 2])
    phi2 = np.array([0, 0.1, -0.1])
    vlos = np.array([10, 15, 12])
    stream = StellarStream(phi1, phi2, vlos, name="TestStream")

    assert np.all(stream.phi1 == phi1)
    assert np.all(stream.phi2 == phi2)
    assert np.all(stream.vlos == vlos)
    assert stream.name == "TestStream"


def test_from_catalog():
    phi1 = [0, 1]
    phi2 = [0.2, -0.2]
    vlos = [5, 6]
    stream = StellarStream.from_catalog(phi1, phi2, vlos, name="CatalogStream")
    assert isinstance(stream, StellarStream)
    assert stream.phi1.shape[0] == 2


# ----------------------------
# Selection
# ----------------------------
def test_selection(simple_stream):
    def criterion_phi1(phi1, phi2, vlos, phi1_lim=(2, 8)):
        return (phi1 > phi1_lim[0]) & (phi1 < phi1_lim[1])

    simple_stream.register_criterion("phi1_cut", criterion_phi1)
    new_stream = simple_stream.select("phi1_cut")

    assert new_stream.phi1.min() > 2
    assert new_stream.phi1.max() < 8
    assert len(new_stream.phi1) <= len(simple_stream.phi1)


# ----------------------------
# Density
# ----------------------------
def test_density_phi1(simple_stream):
    x, dens = simple_stream.density_phi1(precision=0.5)
    assert len(x) == len(dens)
    assert np.all(dens >= 0)
    dx = x[1] - x[0]
    np.testing.assert_allclose(np.sum(dens * dx), 1.0, rtol=1e-6)


# ----------------------------
# Velocity dispersion
# ----------------------------
def test_velocity_dispersion(simple_stream):
    x, sigma = simple_stream.velocity_dispersion(bins=10)
    assert len(x) == len(sigma)
    assert not np.any(np.isnan(x))


# ----------------------------
# Power spectrum
# ----------------------------
def test_power_spectrum(constant_stream):
    freqs, ps = constant_stream.power_spectrum(precision=0.5)
    assert len(freqs) == len(ps)
    assert np.all(ps >= 0)


# ----------------------------
# Denoising and detrending
# ----------------------------
def test_gaussian_detrend(simple_stream):
    detrended = simple_stream.run_analysis("gaussian_detrend", smoothing_scale=0.5)
    assert len(detrended.phi1) == len(simple_stream.phi1)


def test_gaussian_process_detrend(simple_stream):
    detrended = simple_stream.run_analysis("gaussian_process_detrend", n_bins=50)
    assert len(detrended.phi1) == len(simple_stream.phi1)


# ----------------------------
# Noise injection
# ----------------------------
def test_with_gaussian_noise(constant_stream):
    noisy = constant_stream.run_analysis(
        "add_gauss_noise", sigma_phi1=0.1, sigma_phi2=0.1, sigma_v=0.5, random_state=42
    )
    assert noisy.phi1.shape[0] == constant_stream.phi1.shape[0]
    assert not np.allclose(noisy.phi1, constant_stream.phi1)
    assert not np.allclose(noisy.phi2, constant_stream.phi2)
    assert not np.allclose(noisy.vlos, constant_stream.vlos)


# ----------------------------
# Plotting smoke tests
# ----------------------------
def test_plot_stream(simple_stream):
    ax = simple_stream.plot_stream()
    assert ax is not None


def test_plot_density(simple_stream):
    StellarStream.plot_density(simple_stream)
    # No assertion needed; test passes if no exception raised


def test_plot_power_spectrum(simple_stream):
    StellarStream.plot_power_spectrum(simple_stream)


def test_plot_power_spectrum_denoised(simple_stream):
    StellarStream.plot_power_spectrum_denoised(simple_stream)
