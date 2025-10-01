from dataclasses import dataclass, field
from typing import Callable, ClassVar, Tuple

import astropy.units as u
import gala.coordinates as galacoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from celerite2 import GaussianProcess, terms
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import periodogram, welch


class hybridmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        """
        When accessed as C.foo, obj is None and cls is the class C.
        When accessed as c.foo, obj is the instance c and cls is C.
        We wrap self.func in a closure so that calling it always
        injects the correct first argument.
        """
        first = obj if obj is not None else cls

        def wrapper(*args, **kwargs):
            return self.func(first, *args, **kwargs)

        return wrapper


@dataclass
class StellarStream:
    # --- Core data arrays -------------------------------------------
    phi1: np.ndarray
    phi2: np.ndarray
    vlos: np.ndarray
    name: str

    # --- Class-level registries for criteria and analyses -----------
    _class_criteria: ClassVar[dict[str, Callable]] = {}
    _class_analyses: ClassVar[dict[str, Callable]] = {}

    # --- Internal cache for expensive results ----------------------
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Instance-specific registries (override or supplement class-level)
        self._criteria: dict[str, Callable] = {}
        self._analyses: dict[str, Callable] = {}

        self.length = self.phi1.max() - self.phi1.min()

    # --- Alternate constructors -------------------------------------
    @classmethod
    def from_simulation(cls, name: str, path_to_data: str) -> "StellarStream":
        """
        Load simulation data from a directory and reshape it for analysis into a StellarStream Class.


        Parameters
        ----------
            path_to_data:
                The directory containing the simulation data files.

        Returns
        -------
            phi1:
                The phi1 values.
            phi2:
                The phi2 values.
            line_of_sight_velocity:
                The line of sight velocity.
        """

        ### Load some data from a simulation file and plot the stream in galactic coordinates
        sim_data = np.load(path_to_data)

        stream_displacements = sim_data["rs3"]
        stream_velocities = sim_data["vs3"]
        # stream_core_displacements = sim_data['rc3'] May be implemented later
        # stream_core_velocities = sim_data['vc3']

        ### Convert the stream data to Galactic coordinates
        simulation_galactic_coordinates = SkyCoord(
            x=stream_displacements[:, 0] * u.kpc,
            y=stream_displacements[:, 1] * u.kpc,
            z=stream_displacements[:, 2] * u.kpc,
            v_x=stream_velocities[:, 0] * (u.km / u.s),
            v_y=stream_velocities[:, 1] * (u.km / u.s),
            v_z=stream_velocities[:, 2] * (u.km / u.s),
            frame="galactocentric",
        )

        ### Transform the coordinates to GD1Koposov10 frame
        sim_st = simulation_galactic_coordinates.transform_to(galacoord.GD1Koposov10)

        phi1 = sim_st.phi1.degree
        phi2 = sim_st.phi2.degree
        line_of_sight_velocity = sim_st.radial_velocity.value

        return cls(phi1, phi2, line_of_sight_velocity, name)

    @classmethod
    def from_catalog(cls, phi1, phi2, vlos, name) -> "StellarStream":
        """
        Wrap pre‐loaded arrays into a StellarStream.

         Parameters
         ----------
             phi1:
                 The phi1 coordinates.
             phi2:
                 The phi2 coordinates.
             vlos:
                 The line of site velocities.
             name:
                 The name of the stream.

         Returns
         -------
             StellarStream:
                 StellarStream class.
        """
        return cls(np.asarray(phi1), np.asarray(phi2), np.asarray(vlos), name)

    # --- Class-level registration methods ---------------------------
    @classmethod
    def register_class_criterion(cls, name: str, func: Callable):
        """Register a criterion available to all instances."""
        cls._class_criteria[name] = func

    @classmethod
    def register_class_analysis(cls, name: str, func: Callable):
        """Register an analysis function available to all instances."""
        cls._class_analyses[name] = func

    # --- Instance-level registration methods ------------------------
    def register_criterion(self, name: str, func: Callable):
        """Register a criterion only on this instance."""
        self._criteria[name] = func

    def register_analysis(self, name: str, func: Callable):
        """Register an analysis only on this instance."""
        self._analyses[name] = func

    # --- Selection using a criterion --------------------------------
    def select(self, criterion: str | Callable, **kwargs) -> "StellarStream":
        """
        Return a new StellarStream filtered by a criterion.

        Parameters
        ----------
        criterion :
          - If str, looks up in instance or class registries
          - If callable, should be func(phi1,phi2,vlos, **kwargs) -> boolean mask
        kwargs : passed to the criterion function

        Returns
        -------
        StellarStream : new object with filtered data
        """
        # Lookup the function
        if isinstance(criterion, str):
            new_name = f"{self.name} | {criterion}"
            func = self._criteria.get(criterion) or self.__class__._class_criteria.get(
                criterion
            )
            if func is None:
                raise ValueError(f"Criterion '{criterion}' not found.")
        else:
            new_name = f"{self.name} | {criterion.__name__}"
            func = criterion

        # Apply and build new instance
        mask = func(self.phi1, self.phi2, self.vlos, **kwargs)
        return StellarStream(
            self.phi1[mask], self.phi2[mask], self.vlos[mask], name=new_name
        )

    # --- Run an analysis by name ------------------------------------
    def run_analysis(self, name: str, **kwargs):
        """
        Execute a registered analysis on this stream.

        Parameters
        ----------
        name : str
          Name of the analysis to run (instance or class)
        kwargs : passed to analysis function

        Returns
        -------
        Whatever the analysis returns (e.g., (x, y) or a Plot)
        """
        func = self._analyses.get(name) or self.__class__._class_analyses.get(name)
        if func is None:
            raise ValueError(f"Analysis '{name}' not found.")
        return func(self, **kwargs)

    def degrees_to_bins(self, degrees):
        """Helper function to convert degrees to bins."""
        return int(round(self.length / degrees))

    # --- Density along φ₁ -------------------------------------------
    def density_phi1(
        self,
        precision: float = 1.0,
        bins: int | None = None,
        use_bins: bool = False,
        phi1_min: float | None = None,
        phi1_max: float | None = None,
        detrend_deg: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to compute the density of stars in a stream along phi1 axis.
        Parameters
        ---------
            precision:
                The precision in degrees for binning.
            use_bins:
                If True, uses the provided bins instead of calculating from precision. False by default.
            bins:
                The number of bins.
            phi1_min:
                The minimum value of phi1.
            phi1_max:
                The maximum value of phi1.
            detrend_deg: int
                Degree of detrending polynomial. 0 or which gives no detrend by default.
        Returns
        -------
            centers:
                The centers of the bins.
            density:
                The density of stars in each bin.
        """

        if not use_bins:
            bins = self.degrees_to_bins(precision)

        key = (
            "dens1",
            int(bins),
            None if phi1_min is None else float(round(phi1_min, 12)),
            None if phi1_max is None else float(round(phi1_max, 12)),
            detrend_deg,
        )

        if key not in self._cache:
            # fall back to min/max of data if not provided
            if phi1_min is None:
                phi1_min = np.min(self.phi1)
            if phi1_max is None:
                phi1_max = np.max(self.phi1)

            edges = np.linspace(phi1_min, phi1_max, bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            counts, _ = np.histogram(self.phi1, bins=edges)

            widths = np.diff(edges)
            dens = counts / widths

            trend = np.polynomial.Polynomial.fit(centers, dens, deg=detrend_deg)
            dens -= trend(centers)

            if min(dens) < 0:
                dens -= min(dens)

            # normalize so integral(dens dx) == 1
            area = np.sum(dens * widths)
            dens = dens / area

            self._cache[key] = (centers, dens)
        return self._cache[key]

    # --- Velocity dispersion in φ₂ bins -----------------------------
    def velocity_dispersion(
        self,
        precision: float = 1.0,
        bins: int = 40,
        phi2_min: float = None,
        phi2_max: float = None,
        sigma_clip: float = 0,
        use_bins=False,
    ):
        """
        Function to compute the velocity dispersion of stars in a stream along phi2.

        Parameters
        -------
            precision:
                The precision in degrees for binning.
            use_bins:
                If True, uses the provided bins instead of calculating from precision. False by default.
            bins:
                The number of bins for the histogram.
            phi2_min:
                The minimum value of phi2.
            phi2_max:
                The maximum value of phi2.
            sigma_clip:
                The sigma clipping threshold. 0 For no clipping.
        Returns
        -------
            centers:
                The centers of the bins.
            sigma_velocity:
                The velocity dispersion of stars in phi1 coordinates.
        """
        if not use_bins:
            bins = self.degrees_to_bins(precision)

        key = ("sigma", bins, phi2_min, phi2_max, sigma_clip)
        if key not in self._cache:
            if phi2_min is None:
                phi2_min = np.min(self.phi2)
            if phi2_max is None:
                phi2_max = np.max(self.phi2)
            edges = np.linspace(phi2_min, phi2_max, bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            sigmas = np.empty_like(centers)
            for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
                mask = (self.phi2 >= lo) & (self.phi2 < hi)
                if not np.any(mask):
                    sigmas[i] = np.nan
                else:
                    data = self.vlos[mask]
                    if sigma_clip > 0:
                        data, _, _ = sigmaclip(data, sigma_clip)
                    sigmas[i] = np.std(data)
            self._cache[key] = (centers, sigmas)
        return self._cache[key]

    # --- Power spectrum of the φ₁‐density ---------------------------
    def power_spectrum(
        self,
        precision: float = 1.0,
        bins: int = 256,
        use_bins=False,
        window="boxcar",
        detrend_deg=0,
        **kwargs,
    ):
        """
        Function to produce the power spectrum of the phi1 densities.
        Parameters
        ----------
            precision:
                The precision in degrees for binning.
            use_bins:
                If True, uses the provided bins instead of calculating from precision. False by default.
            bins: int
                The number of bins to use for the densities.
            window: None or int
                Window size or None for no window.
        Returns
        -------
            frequencies: array
                The frequencies.
            power_spectrum: array
                The power spectral densities.
        """

        if not use_bins:
            bins = self.degrees_to_bins(precision)

        key = ("ps", bins, window, detrend_deg, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            x, dens = self.density_phi1(bins=bins, use_bins=True, **kwargs)
            fs = 1.0 / (x[1] - x[0])

            freqs, ps = periodogram(dens, fs=fs, window=window)
            # ps *= len(self.phi1) # scale by number of points to get power instead of power density

            self._cache[key] = (freqs, ps)
        return self._cache[key]

    def power_spectrum_denoised(
        self,
        precision: float = 1.0,
        bins: int = 256,
        use_bins: bool = False,
        tukey_alpha: float = 0.25,
        n_segments: int = 6,
        overlap: float = 0.5,
        subtract_highk_floor: bool = True,
        **kwargs,
    ):
        """
        Compute a denoised 1D power spectrum of the stream density.

        Parameters
        ----------
        precision : float
            Bin width in degrees (if use_bins=False).
        bins : int
            Number of bins (if use_bins=True).
        use_bins : bool
            If True, 'bins' sets the number of bins instead of precision.
        detrend : str
            Detrending method: "none", "poly2", "poly3".
        window : str
            Window to reduce edge leakage: "tukey", "hann".
        tukey_alpha : float
            Alpha parameter for Tukey window.
        method : str
            Spectrum method: "fft" (single segment) or "welch" (averaged).
        n_segments : int
            Number of segments for Welch method.
        overlap : float
            Fractional overlap for Welch (0–1).
        subtract_highk_floor : bool
            If True, subtract empirical white-noise floor at high frequencies.

        Returns
        -------
        f : ndarray
            Frequencies in cycles per degree.
        ps : ndarray
            Power spectral density.
        """

        if not use_bins:
            bins = self.degrees_to_bins(precision)

        # --- caching key
        key = (
            "ps_denoised",
            precision,
            bins,
            use_bins,
            tukey_alpha,
            n_segments,
            overlap,
            subtract_highk_floor,
        )
        if key in self._cache:
            return self._cache[key]

        x, dens = self.density_phi1(bins=bins, **kwargs)
        fs = 1.0 / (x[1] - x[0])

        nperseg = int(np.floor(len(x) / (1 + (1 - overlap) * (n_segments - 1))))
        nperseg = max(nperseg, 32)
        noverlap = int(overlap * nperseg)
        freqs, ps = welch(
            dens,
            fs=fs,
            window=("tukey", tukey_alpha),
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            return_onesided=True,
            scaling="density",
        )

        # --- subtract white noise floor
        if subtract_highk_floor:
            hi = freqs > 0.7 * np.max(freqs)
            if np.any(hi):
                floor = np.median(ps[hi])
                ps = np.clip(ps - floor, a_min=0.0, a_max=None)

        self._cache[key] = (freqs, ps)

        return freqs, ps

    # --- Plotting utilities -----------------------------------------
    def plot_stream(self, ax=None):
        """Simple φ₂ vs φ₁ + vlos vs φ₁ plot."""

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].scatter(self.phi1, self.phi2, s=0.5)
        ax[0].set_ylabel(r"$\phi_2$ [deg]")
        ax[1].scatter(self.phi1, self.vlos, s=0.5)
        ax[1].set_ylabel(r"$v_{\rm los}$ [km/s]")
        ax[1].set_xlabel(r"$\phi_1$ [deg]")
        ax[0].set_title(f"Stellar Stream: {self.name}")
        return ax

    @hybridmethod
    def plot_density(self_or_cls, *streams, **kwargs):
        """
        Plot density for multiple StellarStream instances if called from the class,
        or for a single instance if called from an instance.
        """
        if isinstance(self_or_cls, type):
            if not streams:
                raise ValueError(
                    "You must provide at least one StellarStream instance."
                )

            plt.figure()
            for stream in streams:
                if not isinstance(stream, StellarStream):
                    raise TypeError(
                        f"Expected StellarStream instance, got {type(stream)}"
                    )
                x, y = stream.density_phi1(**kwargs)
                plt.plot(x, y, label=f"Stream {stream.name}")
            plt.xlabel(r"$\phi_1$ [deg]")
            plt.ylabel("Density (stars/deg)")
            plt.legend()
            plt.title("Density of Multiple Stellar Streams")
            plt.show()
        else:
            x, y = self_or_cls.density_phi1(**kwargs)
            plt.figure()
            plt.plot(x, y, label="Stream")
            plt.xlabel(r"$\phi_1$ [deg]")
            plt.ylabel("Density (stars/deg)")
            plt.title(f"Density of Stellar Stream {self_or_cls.name}")
            plt.legend()
            plt.show()

    @hybridmethod
    def plot_power_spectrum(self_or_cls, *streams, **kwargs):
        """
        Plot power spectrum for multiple StellarStream instances if called from the class,
        or for a single instance if called from an instance.
        """
        if isinstance(self_or_cls, type):
            if not streams:
                raise ValueError(
                    "You must provide at least one StellarStream instance."
                )

            plt.figure()
            for stream in streams:
                if not isinstance(stream, StellarStream):
                    raise TypeError(
                        f"Expected StellarStream instance, got {type(stream)}"
                    )
                freqs, ps = stream.power_spectrum(**kwargs)
                plt.loglog(
                    freqs[1:], ps[1:], label=f"Stream {stream.name}"
                )  # Zero freq skews log graphx
            plt.xlabel("Frequency (1/deg)")
            plt.ylabel("Power")
            plt.legend()
            plt.title("Power Spectrum of Multiple Stellar Streams")
            plt.show()
        else:
            freqs, ps = self_or_cls.power_spectrum(**kwargs)
            plt.figure()
            plt.loglog(freqs[1:], ps[1:])  # Zero freq skews log graph
            plt.xlabel("Frequency (1/deg)")
            plt.ylabel("Power")
            plt.title(f"Power Spectrum of {self_or_cls.name}")

    @hybridmethod
    def plot_power_spectrum_denoised(self_or_cls, *streams, **kwargs):
        """
        Plot power spectrum for multiple StellarStream instances if called from the class,
        or for a single instance if called from an instance.
        """
        if isinstance(self_or_cls, type):
            if not streams:
                raise ValueError(
                    "You must provide at least one StellarStream instance."
                )

            plt.figure()
            for stream in streams:
                if not isinstance(stream, StellarStream):
                    raise TypeError(
                        f"Expected StellarStream instance, got {type(stream)}"
                    )
                freqs, ps = stream.power_spectrum_denoised(**kwargs)
                plt.loglog(
                    freqs[1:], ps[1:], label=f"Stream {stream.name}"
                )  # Zero freq skews log graphx
            plt.xlabel("Frequency (1/deg)")
            plt.ylabel("Power")
            plt.legend()
            plt.title("Denoised Power Spectrum of Multiple Stellar Streams")
            plt.show()
        else:
            freqs, ps = self_or_cls.power_spectrum_denoised(**kwargs)
            plt.figure()
            plt.loglog(freqs[1:], ps[1:])  # Zero freq skews log graph
            plt.xlabel("Frequency (1/deg)")
            plt.ylabel("Power")
            plt.title(f"Denoised Power Spectrum of {self_or_cls.name}")


# --- Class-level criteria for selection ---------------------------


def restrict_phi1(phi1, phi2, vlos, phi1_lim: Tuple[float, float] = (-80, 10)):
    # select only particles between phi1_beginning and phi1_end
    stream_beginning_cutoff = phi1 > phi1_lim[0]
    stream_ending_cutoff = phi1 < phi1_lim[1]
    return stream_beginning_cutoff & stream_ending_cutoff


def restrict_phi2(phi1, phi2, vlos, phi2_lim=1.0):
    # select only particles within |phi2| < phi2_lim
    return np.abs(phi2) < phi2_lim


# --- Class-level criteria for analysis ---------------------------


def detrend_stream(self, polynomial_fit_degree: int = 4):
    ### Detrend phi2
    phi1_phi2_fit = np.polynomial.polynomial.Polynomial.fit(
        self.phi1, self.phi2, polynomial_fit_degree
    )
    phi2_detrend = self.phi2 - phi1_phi2_fit(self.phi1)

    ### Detrend line of sight velocity
    phi1_vel_fit = np.polynomial.polynomial.Polynomial.fit(
        self.phi1, self.vlos, polynomial_fit_degree
    )
    line_of_sight_velocity_detrend = self.vlos - phi1_vel_fit(self.phi1)
    return StellarStream(
        self.phi1, phi2_detrend, line_of_sight_velocity_detrend, name=self.name
    )


def detrend_stream_phi1(self, polynomial_fit_degree: int = 4):
    ### Detrend phi1
    phi1_phi2_fit = np.polynomial.polynomial.Polynomial.fit(
        self.phi2, self.phi1, polynomial_fit_degree
    )
    phi1_detrend = self.phi1 - phi1_phi2_fit(self.phi2)

    ### Detrend line of sight velocity
    phi2_vel_fit = np.polynomial.polynomial.Polynomial.fit(
        self.phi2, self.vlos, polynomial_fit_degree
    )
    line_of_sight_velocity_detrend = self.vlos - phi2_vel_fit(self.phi2)
    return StellarStream(
        phi1_detrend, self.phi2, line_of_sight_velocity_detrend, name=self.name
    )


def gaussian_detrend(self, smoothing_scale: float = 0.1) -> StellarStream:
    """
    Detrend phi2 and vlos in `stream` using a Gaussian kernel applied along phi1.

    Parameters
    ----------
    smoothing_scale : float
        Smoothing scale in the same units as `phi1` (not pixels). The function converts
        it to pixels by dividing by the median spacing in `phi1`.

    Returns
    -------
    Stellar Stream  (and optionally phi2_trend, vlos_trend)
        With detrended arrays in the original order/indexing of stream.phi2 and stream.vlos.
    """

    # convert to numpy arrays for computation but keep originals for re-wrapping
    phi1_orig = self.phi1
    phi2_orig = self.phi2
    vlos_orig = self.vlos

    x = np.asarray(phi1_orig, dtype=float)
    y_phi2 = np.asarray(phi2_orig, dtype=float)
    y_vlos = np.asarray(vlos_orig, dtype=float)

    if x.size == 0:
        raise ValueError("phi1 is empty")

    # sort
    order = np.argsort(x)
    x_sorted = x[order]
    phi2_sorted = y_phi2[order]
    vlos_sorted = y_vlos[order]

    # compute representative dx (ignore zero diffs from duplicates)
    diffs = np.diff(x_sorted)
    nonzero = diffs[np.abs(diffs) > 0]
    if nonzero.size == 0:
        raise ValueError(
            "phi1 has no spacing (all values equal). Cannot compute smoothing in pixel units."
        )
    dx = np.median(nonzero)

    # convert smoothing scale (same units as phi1) to pixels for gaussian_filter1d
    sigma_pix = float(smoothing_scale) / float(dx)
    if sigma_pix <= 0:
        # sigma=0 means no smoothing; gaussian_filter1d with sigma=0 returns original array
        sigma_pix = 0.0

    # apply Gaussian smoothing on the sorted data (reflect mode to reduce edge artifacts)
    # gaussian_filter1d accepts sigma==0 (no smoothing)
    phi2_trend_sorted = gaussian_filter1d(phi2_sorted, sigma_pix, mode="reflect")
    vlos_trend_sorted = gaussian_filter1d(vlos_sorted, sigma_pix, mode="reflect")

    # map the trends back to original order
    inverse_order = np.argsort(order)
    phi2_trend_orig = phi2_trend_sorted[inverse_order]
    vlos_trend_orig = vlos_trend_sorted[inverse_order]

    # detrend in original order
    phi2_detrend = np.asarray(phi2_orig, dtype=float) - phi2_trend_orig
    vlos_detrend = np.asarray(vlos_orig, dtype=float) - vlos_trend_orig

    return StellarStream(self.phi1, phi2_detrend, vlos_detrend, name=self.name)


def gaussian_process_detrend(self, n_bins):
    """
    Detrend the stream using a Gaussian process.
    """

    def bin_data(x, y, n_bins=2000):
        # x sorted assumed
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        idx = np.digitize(x, edges) - 1
        idx[idx < 0] = 0
        idx[idx >= n_bins] = n_bins - 1
        xb = 0.5 * (edges[:-1] + edges[1:])

        # sums for mean and variance
        ysum = np.bincount(idx, weights=y, minlength=n_bins)
        y2sum = np.bincount(idx, weights=y * y, minlength=n_bins)
        count = np.bincount(idx, minlength=n_bins)

        with np.errstate(invalid="ignore", divide="ignore"):
            ymean = ysum / count
            yvar = (y2sum / count) - (ymean**2)

        # Mask empty bins
        mask = count > 0
        return xb[mask], ymean[mask], count[mask], yvar[mask]

    def fit_gp_and_predict(xb, yb, counts, yvar, phi1_positions):
        """
        Fit a celerite2 GP on binned (xb, yb) with per-bin variance yvar and counts,
        then predict GP mean at phi1_positions (unbinned positions).
        Returns mu_full, var_full.
        """
        # per-bin uncertainty (std of the mean): sqrt(var) / sqrt(N)
        # protect against negative/NaN variances and small counts
        var_floor = 1e-8 * np.nanmax(np.abs(yb)) ** 2 + 1e-12
        per_bin_var = np.where(
            np.isfinite(yvar), np.maximum(yvar, var_floor), var_floor
        )
        yerr_binned = np.sqrt(per_bin_var) / np.sqrt(np.maximum(counts, 1))

        # initial guesses
        amp0 = max(np.std(yb[np.isfinite(yb)]), 1e-6)
        stream_length = xb.max() - xb.min() if (xb.max() > xb.min()) else 1.0
        dx_med = np.median(np.diff(xb)) if len(xb) > 1 else stream_length / 100.0
        rho0 = max(stream_length / 8.0, 3.0 * dx_med, 1e-3)
        jitter0 = max(1e-3 * amp0, 1e-6)

        # Build GP
        term_init = terms.SHOTerm(sigma=amp0, rho=rho0, Q=1.0)
        gp_local = GaussianProcess(term_init, mean=np.median(yb))
        gp_local.compute(xb, diag=(yerr_binned**2 + jitter0))

        def set_params(params, gp_obj):
            sigma = np.exp(params[0])
            rho = np.exp(params[1])
            jitter = np.exp(params[2])
            gp_obj.mean = np.median(yb)  # keep mean fixed
            gp_obj.kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1.0)
            gp_obj.compute(xb, diag=(yerr_binned**2 + jitter), quiet=True)
            return gp_obj

        def neg_log_like(params):
            g = set_params(params, gp_local)
            return -g.log_likelihood(yb)

        p0 = np.array([np.log(amp0), np.log(rho0), np.log(jitter0)])
        sol = minimize(neg_log_like, p0, method="L-BFGS-B", options={"maxiter": 200})

        # set gp with fitted params
        gp_fitted = set_params(sol.x, gp_local)

        # predict at full resolution (phi1 positions)
        mu_full, var_full = gp_fitted.predict(yb, t=phi1_positions, return_var=True)
        return mu_full, var_full

    # prepare sorted data
    phi1 = np.asarray(self.phi1, dtype=float)
    phi2 = np.asarray(self.phi2, dtype=float)
    vlos = np.asarray(self.vlos, dtype=float)
    mask_all = np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(vlos)
    phi1 = phi1[mask_all]
    phi2 = phi2[mask_all]
    vlos = vlos[mask_all]

    # sort by phi1 and apply same order to phi2 and vlos
    order = np.argsort(phi1)
    phi1 = phi1[order]
    phi2 = phi2[order]
    vlos = vlos[order]

    # bin both phi2 and vlos using the same bins (so xb is consistent)
    xb, phi2_binned, counts_phi2, var_phi2 = bin_data(phi1, phi2, n_bins=n_bins)
    _, vlos_binned, counts_vlos, var_vlos = bin_data(phi1, vlos, n_bins=n_bins)

    # If binning produced no bins (empty), just return original stream
    if len(xb) == 0:
        return StellarStream(self.phi1, self.phi2, self.vlos, name=self.name)

    # Fit GP & predict for phi2
    mu_phi2_full, _ = fit_gp_and_predict(xb, phi2_binned, counts_phi2, var_phi2, phi1)
    phi2_detrend = phi2 - mu_phi2_full

    # Fit GP & predict for vlos
    mu_vlos_full, _ = fit_gp_and_predict(xb, vlos_binned, counts_vlos, var_vlos, phi1)
    vlos_detrend = vlos - mu_vlos_full

    # return new StellarStream with detrended phi2 and vlos (phi1 kept original ordering)
    return StellarStream(self.phi1, phi2_detrend, vlos_detrend, name=self.name)


def with_gaussian_noise(
    self,
    sigma_phi1: float = 0.01,
    sigma_phi2: float = 0.01,
    sigma_v: float = 1.0,
    random_state: int | None = None,
) -> "StellarStream":
    # Add Gaussian noise to the stream data.
    rng = np.random.default_rng(random_state)
    phi1_n = self.phi1 + rng.normal(0, sigma_phi1, size=self.phi1.shape)
    phi2_n = self.phi2 + rng.normal(0, sigma_phi2, size=self.phi2.shape)
    vlos_n = self.vlos + rng.normal(0, sigma_v, size=self.vlos.shape)
    return StellarStream(phi1_n, phi2_n, vlos_n, name=f"{self.name} + Noise")


# --- Register class-level criteria and analyses -------------------
StellarStream.register_class_criterion("restrict_phi1", restrict_phi1)
StellarStream.register_class_criterion("restrict_phi2", restrict_phi2)

StellarStream.register_class_analysis("detrend", detrend_stream)
StellarStream.register_class_analysis("detrend_phi1", detrend_stream_phi1)
StellarStream.register_class_analysis("gaussian_detrend", gaussian_detrend)
StellarStream.register_class_analysis(
    "gaussian_process_detrend", gaussian_process_detrend
)
StellarStream.register_class_analysis("add_gauss_noise", with_gaussian_noise)
