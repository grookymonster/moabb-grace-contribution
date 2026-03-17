"""Neural signature visualisation for MOABB datasets.

Generates interactive Plotly figures showing the key neurophysiological
signature for each BCI paradigm:

- **Motor Imagery** -- ERD/ERS time-frequency maps
- **P300 / ERP** -- Evoked response waveforms (Target vs NonTarget)
- **SSVEP** -- Power spectrum / SNR at stimulus frequencies
- **c-VEP** -- Code-modulated evoked response + PSD
- **Resting State** -- Band power PSD comparison across conditions

Public API
----------
generate_neural_signature(dataset, subjects=None, output_dir=None)
    High-level entry point: generates all HTML files for a dataset.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

import moabb
from moabb.analysis.style import (
    GRID_COLOR,
    MOABB_AMBER,
    MOABB_CORAL,
    MOABB_DARK_TEXT,
    MOABB_NAVY,
    MOABB_PALETTE,
    MOABB_PURPLE,
    MOABB_SKY,
    MOABB_TEAL,
    _DEFAULT_SOURCE,
)


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Plotly import
# ---------------------------------------------------------------------------


def _check_plotly():
    """Raise a helpful error if plotly is not installed."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "plotly is required for neural signature figures. "
            "Install with: pip install moabb[interactive]"
        )


# ---------------------------------------------------------------------------
# Plotly style adapter  (MOABB Editorial — Economist-inspired)
# ---------------------------------------------------------------------------

# Font stack: Georgia for the distinctive serif scientific feel,
# with system fallbacks.  Matches the matplotlib "serif" choice.
_FONT_FAMILY = "Georgia, Cambria, 'Times New Roman', serif"

# Semantic color order for interactive plots: navy, coral, teal, purple, amber, sky.
_PLOT_PALETTE = [MOABB_NAVY, MOABB_CORAL, MOABB_TEAL, MOABB_PURPLE, MOABB_AMBER, MOABB_SKY]

# Subtle background tint — avoids the clinical pure-white look
_PAPER_BG = "#FAFBFC"
_PLOT_BG = "#FAFBFC"

# Lighter grid that won't compete with heatmaps
_GRID_LIGHT = "rgba(117, 141, 153, 0.18)"


def get_plotly_template():
    """Return a Plotly layout template matching the MOABB visual identity.

    Mirrors the Economist-inspired style from :func:`apply_moabb_style`:
    navy accent line at top, teal pip, serif typography, minimal grid.
    """
    import plotly.graph_objects as go

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(
            family=_FONT_FAMILY,
            color=MOABB_DARK_TEXT,
            size=12,
        ),
        title=dict(
            font=dict(size=20, color=MOABB_DARK_TEXT, family=_FONT_FAMILY),
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(b=12),
        ),
        colorway=MOABB_PALETTE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        margin=dict(l=72, r=24, t=80, b=64),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=GRID_COLOR,
            linewidth=0.7,
            ticks="outside",
            ticklen=4,
            tickwidth=0.7,
            tickcolor=GRID_COLOR,
            title_font=dict(size=13, color=MOABB_DARK_TEXT),
            title_standoff=10,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID_LIGHT,
            gridwidth=0.5,
            griddash="solid",
            zeroline=False,
            showline=False,
            ticks="",
            title_font=dict(size=13, color=MOABB_DARK_TEXT),
            title_standoff=8,
        ),
        legend=dict(
            bgcolor="rgba(250, 251, 252, 0.92)",
            borderwidth=0,
            font=dict(size=11, family=_FONT_FAMILY),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=MOABB_NAVY,
            font=dict(size=12, family=_FONT_FAMILY, color=MOABB_DARK_TEXT),
        ),
    )
    return template


def get_plotly_colorscale():
    """Return a diverging blue-gray-red colorscale for ERD/ERS maps.

    Uses the standard neuroscience convention: blue = desynchronisation
    (ERD), red = synchronisation (ERS), with a neutral light-gray
    midpoint that avoids the washed-out look of pure white.
    """
    return [
        [0.00, "#2166AC"],  # deep blue — strong ERD
        [0.25, "#67A9CF"],  # mid blue
        [0.50, "#F0F0F0"],  # neutral warm gray — NOT white
        [0.75, "#EF8A62"],  # warm salmon
        [1.00, "#B2182B"],  # deep red — strong ERS
    ]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r, g, b, a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _get_montage_xy(ch_names: list[str]) -> dict[str, tuple[float, float]]:
    """Return normalised 2-D (x, y) positions for *ch_names* from 10-20."""
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_3d = montage.get_positions()["ch_pos"]
    # Collect raw x, y for channels present in the montage
    raw: dict[str, tuple[float, float]] = {}
    for ch in ch_names:
        if ch in pos_3d:
            raw[ch] = (pos_3d[ch][0], pos_3d[ch][1])
    if not raw:
        return {}
    xs = [v[0] for v in raw.values()]
    ys = [v[1] for v in raw.values()]
    # Normalise to head-radius (use the full montage extent)
    all_xy = np.array([[pos_3d[c][0], pos_3d[c][1]]
                       for c in montage.ch_names if c in pos_3d])
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    radius = np.max(np.sqrt((all_xy[:, 0] - cx) ** 2 + (all_xy[:, 1] - cy) ** 2))
    if radius == 0:
        radius = 1.0
    return {ch: ((x - cx) / radius, (y - cy) / radius) for ch, (x, y) in raw.items()}


def _add_head_inset(
    fig,
    ch_names: list[str],
    active_idx: int = 0,
    xdomain: tuple[float, float] = (0.88, 1.0),
    ydomain: tuple[float, float] = (0.0, 0.18),
    ax_id: int = 99,
) -> int:
    """Draw a small scalp head with interactive electrode dots.

    Uses a secondary axes so that electrode highlights can be toggled
    by the channel slider.  Returns the number of highlight traces added
    (one per channel) so the caller can wire up visibility.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    ch_names : list of str
        Channels to plot.
    active_idx : int
        Index of the initially active channel.
    xdomain, ydomain : tuple
        Paper-coordinate domain for the inset axes.
    ax_id : int
        Axis number suffix for the secondary axes.

    Returns
    -------
    int
        Number of highlight traces appended (== len(ch_names)).
    """
    import plotly.graph_objects as go

    positions = _get_montage_xy(ch_names)
    if not positions:
        return 0

    xax = f"x{ax_id}"
    yax = f"y{ax_id}"
    xref = f"x{ax_id}"
    yref = f"y{ax_id}"

    # Add hidden secondary axes in the corner
    fig.update_layout(**{
        f"xaxis{ax_id}": dict(
            domain=list(xdomain), anchor=yax.replace("y", "y"),
            range=[-1.4, 1.4], showgrid=False, zeroline=False,
            showticklabels=False, showline=False,
        ),
        f"yaxis{ax_id}": dict(
            domain=list(ydomain), anchor=xax.replace("x", "x"),
            range=[-1.4, 1.4], showgrid=False, zeroline=False,
            showticklabels=False, showline=False,
            scaleanchor=xax, scaleratio=1,
        ),
    })

    # Head outline
    fig.add_shape(
        type="circle", xref=xref, yref=yref,
        x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color=MOABB_NAVY, width=1.5),
        fillcolor="rgba(250,251,252,0.9)",
    )
    # Nose
    fig.add_shape(
        type="path", xref=xref, yref=yref,
        path="M -0.12 1.0 L 0 1.22 L 0.12 1.0",
        line=dict(color=MOABB_NAVY, width=1.5),
    )
    # Ears
    for s in (-1, 1):
        fig.add_shape(
            type="path", xref=xref, yref=yref,
            path=f"M {s*1.0} -0.25 Q {s*1.22} 0 {s*1.0} 0.25",
            line=dict(color=MOABB_NAVY, width=1.5),
        )

    # All electrode dots (always visible, small navy)
    all_x = [positions[ch][0] * 0.85 for ch in ch_names if ch in positions]
    all_y = [positions[ch][1] * 0.85 for ch in ch_names if ch in positions]
    all_labels = [ch for ch in ch_names if ch in positions]
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y, mode="markers",
        marker=dict(size=6, color=MOABB_NAVY, opacity=0.35),
        hoverinfo="text", hovertext=all_labels,
        showlegend=False,
        xaxis=xax, yaxis=yax,
    ))

    # Per-channel highlight traces (one per channel, toggled by slider)
    for ch_i, ch in enumerate(ch_names):
        if ch not in positions:
            # Still add a dummy trace to keep indexing consistent
            fig.add_trace(go.Scatter(
                x=[0], y=[0], mode="markers",
                marker=dict(size=0, opacity=0),
                hoverinfo="skip", showlegend=False,
                visible=False, xaxis=xax, yaxis=yax,
            ))
            continue
        px, py = positions[ch][0] * 0.85, positions[ch][1] * 0.85
        fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers+text",
            marker=dict(size=18, color=MOABB_CORAL, symbol="circle",
                        line=dict(width=2.5, color="white")),
            text=[f"<b>{ch}</b>"], textposition="bottom center",
            textfont=dict(size=11, color=MOABB_CORAL, family=_FONT_FAMILY),
            hoverinfo="text", hovertext=ch,
            showlegend=False,
            visible=(ch_i == active_idx),
            xaxis=xax, yaxis=yax,
        ))

    return len(ch_names)


def _build_event_label_map(dataset) -> dict[str, str]:
    """Build a mapping from integer-string event codes to readable names.

    When a paradigm relabels events like ``{"left_hand": 1}`` to
    ``{"1": 1}``, we lose the human-readable names.  This function
    inverts ``dataset.event_id`` so that ``"1"`` maps back to
    ``"left_hand"``.
    """
    label_map: dict[str, str] = {}
    if not hasattr(dataset, "event_id") or not dataset.event_id:
        return label_map
    for name, code in dataset.event_id.items():
        str_code = str(code)
        # Only map if the code looks like an integer string
        if str_code not in label_map:
            label_map[str_code] = name
    return label_map


def _display_name(name: str) -> str:
    """Make an event name human-readable: 'left_hand' -> 'Left Hand'."""
    return name.replace("_", " ").title()


def _relabel_signature(sig: "NeuralSignatureData", label_map: dict[str, str]):
    """Replace integer-string event keys with human-readable names in-place."""
    if not label_map:
        return

    # Build a combined map: int-string -> display name
    display_map = {k: _display_name(v) for k, v in label_map.items()}

    def _remap_dict(d: dict) -> dict:
        return {display_map.get(k, k): v for k, v in d.items()}

    if "evokeds" in sig.data:
        sig.data["evokeds"] = _remap_dict(sig.data["evokeds"])
    if "sems" in sig.data:
        sig.data["sems"] = _remap_dict(sig.data["sems"])
    if "tfr" in sig.data:
        sig.data["tfr"] = _remap_dict(sig.data["tfr"])
    if "psd" in sig.data and isinstance(sig.data["psd"], dict):
        sig.data["psd"] = _remap_dict(sig.data["psd"])
    if "snr" in sig.data:
        sig.data["snr"] = _remap_dict(sig.data["snr"])
    if "band_powers" in sig.data:
        sig.data["band_powers"] = _remap_dict(sig.data["band_powers"])
    if "event_names" in sig.data:
        sig.data["event_names"] = [
            display_map.get(e, e) for e in sig.data["event_names"]
        ]
    if "n_trials" in sig.metadata:
        sig.metadata["n_trials"] = _remap_dict(sig.metadata["n_trials"])


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class NeuralSignatureData:
    """Container for computed neural signature data."""

    paradigm: str  # "imagery", "p300", "ssvep", "cvep", "rstate"
    dataset_name: str
    dataset_code: str
    signature_type: str  # "erd_ers", "erp", "psd_snr", "cvep_response", "rstate_psd"
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------


def _compute_evokeds_and_sems(
    epochs: mne.Epochs,
    event_names: list[str],
) -> tuple[dict, dict, dict]:
    """Compute evoked averages and 95% CI SEMs per event.

    Returns ``(evokeds, sems, n_trials)`` dicts keyed by event name.
    """
    evokeds: dict[str, np.ndarray] = {}
    sems: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n = len(ep)
        n_trials[name] = n
        if n == 0:
            continue
        evk = ep.average()
        evokeds[name] = evk.data
        if n > 1:
            std = np.std(ep.get_data(), axis=0, ddof=1)
            sems[name] = std / np.sqrt(n) * 1.96
        else:
            sems[name] = np.zeros_like(evk.data)
    return evokeds, sems, n_trials


def _compute_psd_per_event(
    epochs: mne.Epochs,
    event_names: list[str],
    fmin: float = 1,
    fmax: float = 50,
    n_fft: int | None = None,
) -> tuple[dict, np.ndarray | None, dict]:
    """Compute mean PSD per event via Welch's method.

    Returns ``(psd_data, freqs, n_trials)`` where *psd_data* maps event
    names to 1-D arrays averaged over epochs and channels.
    """
    if n_fft is None:
        n_fft = len(epochs.times)
    psd_data: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    freqs = None
    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n_trials[name] = len(ep)
        if len(ep) == 0:
            continue
        spectrum = ep.compute_psd(
            method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False
        )
        psd_data[name] = spectrum.get_data().mean(axis=0).mean(axis=0)
        freqs = spectrum.freqs
    return psd_data, freqs, n_trials


# ---------------------------------------------------------------------------
# Paradigm-specific computation functions
# ---------------------------------------------------------------------------


def compute_erp_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
) -> NeuralSignatureData:
    """Compute P300/ERP signature from epochs.

    Averages time-locked trials per event class and computes 95 %
    confidence intervals via the standard error of the mean.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with "Target" and "NonTarget" (or custom) events.
    event_names : list of str or None
        Event names to compare.  Defaults to ["Target", "NonTarget"].

    Returns
    -------
    NeuralSignatureData

    See Also
    --------
    :func:`plot_erp_interactive` : Plot the computed ERP signature.
    :func:`generate_neural_signature` : High-level entry point.

    Notes
    -----
    Follows the evoked-response approach described in the MNE
    visualization tutorial [1]_.

    References
    ----------
    .. [1] https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html
    """
    if event_names is None:
        event_names = [e for e in ["Target", "NonTarget"] if e in epochs.event_id]
    if not event_names:
        event_names = list(epochs.event_id.keys())[:2]

    evokeds, sems, n_trials = _compute_evokeds_and_sems(epochs, event_names)

    return NeuralSignatureData(
        paradigm="p300",
        dataset_name="",
        dataset_code="",
        signature_type="erp",
        data=dict(
            evokeds=evokeds,
            sems=sems,
            times=epochs.times,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def _select_central_channels(ch_names: list[str]) -> list[str]:
    """Pick the best available central channels for motor imagery."""
    preferred = ["C3", "C4", "Cz", "C1", "C2", "FC3", "FC4", "CP3", "CP4"]
    selected = [ch for ch in preferred if ch in ch_names]
    if not selected:
        selected = ch_names[:min(3, len(ch_names))]
    return selected


def compute_erd_ers_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
    freqs: np.ndarray | None = None,
    baseline: tuple[float, float] | None = None,
) -> NeuralSignatureData:
    """Compute ERD/ERS time-frequency maps for motor imagery.

    Uses the modern ``epochs.compute_tfr(method="multitaper")`` API
    with ``apply_baseline(mode="percent")`` following the MNE ERDS
    maps tutorial.

    Parameters
    ----------
    epochs : mne.Epochs
        Motor imagery epochs.
    event_names : list of str or None
        Event class names.  Defaults to all available.
    freqs : ndarray or None
        Frequencies for TFR.  Defaults to 2-36 Hz in 1 Hz steps
        (matching the MNE ERDS tutorial range).
    baseline : tuple or None
        Baseline interval ``(tmin, tmax)`` in seconds.  Defaults to
        ``(epochs.tmin, 0)``.

    Returns
    -------
    NeuralSignatureData

    See Also
    --------
    :func:`plot_erd_ers_interactive` : Plot the computed ERD/ERS maps.
    :func:`generate_neural_signature` : High-level entry point.

    Notes
    -----
    ``n_cycles`` is set to ``freqs`` (one cycle per Hz), matching the
    MNE ERDS tutorial [1]_.  Baseline correction uses ``mode="percent"``
    so that values represent percentage change from baseline
    (negative = ERD, positive = ERS).

    References
    ----------
    .. [1] https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
    """
    if event_names is None:
        event_names = list(epochs.event_id.keys())
    if freqs is None:
        freqs = np.arange(2, 36, 1.0)
    if baseline is None:
        t0 = epochs.times[0]
        if t0 < 0:
            baseline = (t0, 0.0)
        else:
            # Absolute times (MOABB convention) — use first 0.5s as baseline
            baseline = (t0, t0 + 0.5)

    central_chs = _select_central_channels(epochs.ch_names)
    picks = mne.pick_channels(epochs.ch_names, central_chs, ordered=True)

    tfr_data = {}
    n_trials = {}
    times = epochs.times  # fallback if no events match

    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n_trials[name] = len(ep)
        if len(ep) == 0:
            continue

        # n_cycles = freqs matches the MNE ERDS tutorial convention
        n_cycles = freqs.copy()
        n_cycles = np.clip(n_cycles, 2, None)

        tfr = ep.compute_tfr(
            method="multitaper",
            freqs=freqs,
            n_cycles=n_cycles,
            picks=picks,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=2,
            verbose=False,
        )
        # Baseline-correct as percentage change (ERD/ERS convention)
        tfr.apply_baseline(baseline=baseline, mode="percent", verbose=False)

        # Average across epochs → (n_channels, n_freqs, n_times)
        avg_data = tfr.average().data * 100  # mode="percent" gives fraction
        tfr_data[name] = avg_data
        times = tfr.times

    return NeuralSignatureData(
        paradigm="imagery",
        dataset_name="",
        dataset_code="",
        signature_type="erd_ers",
        data=dict(
            tfr=tfr_data,
            freqs=freqs,
            times=times,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=central_chs,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def _snr_spectrum(
    psd: np.ndarray,
    noise_n_neighbor_freqs: int = 3,
    noise_skip_neighbor_freqs: int = 1,
) -> np.ndarray:
    """Compute SNR spectrum using the MNE SSVEP tutorial approach.

    Divides each frequency bin by the mean power of surrounding bins,
    skipping immediately adjacent bins to avoid spectral leakage.

    Parameters
    ----------
    psd : ndarray, shape (..., n_freqs)
        Power spectral density array.
    noise_n_neighbor_freqs : int
        Number of neighboring frequency bins on each side used to
        estimate noise.
    noise_skip_neighbor_freqs : int
        Number of bins directly adjacent to the target bin to skip.

    Returns
    -------
    ndarray, same shape as *psd*
        SNR values (ratio, not dB).

    References
    ----------
    .. [1] https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
    """
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs),
    ))
    averaging_kernel /= averaging_kernel.sum()

    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"),
        axis=-1, arr=psd,
    )
    edge = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge, edge)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width,
                        constant_values=np.nan)

    return psd / mean_noise


def compute_ssvep_signature(
    epochs: mne.Epochs,
    stimulus_frequencies: list[float] | None = None,
) -> NeuralSignatureData:
    """Compute SSVEP PSD and SNR at stimulus frequencies.

    Uses a single Welch window spanning the full epoch for maximum
    frequency resolution, and computes SNR using the convolution-kernel
    approach from the MNE SSVEP tutorial.

    Parameters
    ----------
    epochs : mne.Epochs
        SSVEP epochs with frequency-labelled events.
    stimulus_frequencies : list of float or None
        Stimulus frequencies in Hz.  Inferred from event names if None.

    Returns
    -------
    NeuralSignatureData

    See Also
    --------
    :func:`plot_ssvep_interactive` : Plot the computed SSVEP spectrum.
    :func:`generate_neural_signature` : High-level entry point.

    Notes
    -----
    Follows the MNE SSVEP tutorial approach [1]_: a single boxcar
    window with ``n_fft`` equal to the epoch length for maximum
    frequency resolution, and SNR computed via a symmetric convolution
    kernel that skips immediately adjacent frequency bins.

    References
    ----------
    .. [1] https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
    """
    event_names = list(epochs.event_id.keys())

    # Infer stimulus frequencies from event names
    if stimulus_frequencies is None:
        stimulus_frequencies = []
        for name in event_names:
            try:
                stimulus_frequencies.append(float(name))
            except ValueError:
                pass

    psd_data = {}
    snr_data = {}
    n_trials = {}
    freqs = None

    # Use full epoch length for max frequency resolution (MNE tutorial)
    n_times = len(epochs.times)
    n_fft = n_times

    for name in event_names:
        if name not in epochs.event_id:
            continue
        ep = epochs[name]
        n_trials[name] = len(ep)
        if len(ep) == 0:
            continue

        spectrum = ep.compute_psd(
            method="welch", fmin=1, fmax=90,
            n_fft=n_fft, n_overlap=0, n_per_seg=None,
            window="boxcar", verbose=False,
        )
        psds, freqs = spectrum.get_data(return_freqs=True)
        # psds: (n_epochs, n_channels, n_freqs)

        psd_mean = psds.mean(axis=(0, 1))  # grand mean -> (n_freqs,)
        psd_data[name] = psd_mean

        # SNR via the MNE convolution-kernel approach
        snr_all = _snr_spectrum(psds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            snr_mean = np.nanmean(snr_all, axis=(0, 1))  # (n_freqs,)

        try:
            target_freq = float(name)
        except ValueError:
            target_freq = None

        if target_freq is not None and freqs is not None:
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            snr_val = snr_mean[freq_idx]
            snr_data[name] = float(snr_val) if np.isfinite(snr_val) else 0.0

    return NeuralSignatureData(
        paradigm="ssvep",
        dataset_name="",
        dataset_code="",
        signature_type="psd_snr",
        data=dict(
            psd=psd_data,
            snr=snr_data,
            freqs=freqs,
            stimulus_frequencies=stimulus_frequencies,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def compute_cvep_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
) -> NeuralSignatureData:
    """Compute c-VEP evoked response and PSD.

    Parameters
    ----------
    epochs : mne.Epochs
        c-VEP epochs with code-level events (e.g. "1.0", "0.0").
    event_names : list of str or None
        Event names.  Defaults to all available.

    Returns
    -------
    NeuralSignatureData

    See Also
    --------
    :func:`plot_cvep_interactive` : Plot the computed c-VEP signature.
    :func:`generate_neural_signature` : High-level entry point.
    """
    if event_names is None:
        event_names = list(epochs.event_id.keys())

    evokeds, sems, n_trials_evk = _compute_evokeds_and_sems(epochs, event_names)
    psd_data, freqs, n_trials_psd = _compute_psd_per_event(
        epochs, event_names, fmin=1, fmax=50,
    )
    # Merge trial counts (evoked loop may have seen different events)
    n_trials = {**n_trials_psd, **n_trials_evk}

    return NeuralSignatureData(
        paradigm="cvep",
        dataset_name="",
        dataset_code="",
        signature_type="cvep_response",
        data=dict(
            evokeds=evokeds,
            sems=sems,
            psd=psd_data,
            freqs=freqs,
            times=epochs.times,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


def compute_rstate_signature(
    epochs: mne.Epochs,
    event_names: list[str] | None = None,
) -> NeuralSignatureData:
    """Compute resting state PSD and band powers per condition.

    Parameters
    ----------
    epochs : mne.Epochs
        Resting state epochs (e.g. eyes_open vs eyes_closed).
    event_names : list of str or None
        Condition names.  Defaults to all available.

    Returns
    -------
    NeuralSignatureData

    See Also
    --------
    :func:`plot_rstate_interactive` : Plot the computed resting state signature.
    :func:`generate_neural_signature` : High-level entry point.

    Notes
    -----
    Band power is computed as relative power (percentage of total
    1-50 Hz power) in delta (1-4), theta (4-8), alpha (8-13),
    beta (13-30), and gamma (30-45) bands.  Follows the global
    field power approach described in the MNE tutorial [1]_.

    References
    ----------
    .. [1] https://mne.tools/stable/auto_examples/time_frequency/time_frequency_global_field_power.html
    """
    if event_names is None:
        event_names = list(epochs.event_id.keys())

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    psd_data, freqs, n_trials = _compute_psd_per_event(
        epochs, event_names, fmin=1, fmax=50,
    )

    # Compute relative band powers from the PSD
    band_powers = {}
    for name, psd in psd_data.items():
        total_power = np.trapezoid(psd, freqs)
        bp = {}
        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            bp[band_name] = np.trapezoid(psd[mask], freqs[mask]) / total_power * 100
        band_powers[name] = bp

    return NeuralSignatureData(
        paradigm="rstate",
        dataset_name="",
        dataset_code="",
        signature_type="rstate_psd",
        data=dict(
            psd=psd_data,
            band_powers=band_powers,
            freqs=freqs,
            bands=bands,
            event_names=event_names,
        ),
        metadata=dict(
            ch_names=epochs.ch_names,
            sfreq=epochs.info["sfreq"],
            n_trials=n_trials,
        ),
    )


# ---------------------------------------------------------------------------
# Plotly figure generators
# ---------------------------------------------------------------------------


def _make_figure(**kwargs):
    """Create a Figure with the MOABB template applied."""
    import plotly.graph_objects as go

    template = get_plotly_template()
    fig = go.Figure(**kwargs)
    fig.update_layout(template=template)
    return fig


def _add_branding(fig, title: str, subtitle: str = ""):
    """Add the MOABB Economist-style branding to a Plotly figure.

    Adds: navy accent bar, teal pip, title/subtitle hierarchy, source line.
    """
    source = _DEFAULT_SOURCE.format(version=moabb.__version__)

    # Navy accent bar at top
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0.0, y0=1.005, x1=1.0, y1=1.015,
        fillcolor=MOABB_NAVY, line_width=0,
        layer="above",
    )
    # Teal pip
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0.0, y0=0.997, x1=0.04, y1=1.005,
        fillcolor=MOABB_TEAL, line_width=0,
        layer="above",
    )
    # Title
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b>"
                + (f"<br><span style='font-size:13px;color:{GRID_COLOR}'>"
                   f"{subtitle}</span>" if subtitle else "")
            ),
            font=dict(size=18, color=MOABB_DARK_TEXT, family=_FONT_FAMILY),
            x=0.0,
            xanchor="left",
            y=0.96,
            yanchor="top",
        ),
    )
    # Source footnote
    fig.add_annotation(
        text=f"<i>{source}</i>",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.0, y=-0.08,
        xanchor="left", yanchor="top",
        font=dict(size=9, color=GRID_COLOR, family=_FONT_FAMILY),
    )


def compute_metrics(sig: NeuralSignatureData) -> dict[str, Any]:
    """Compute objective quality metrics for a neural signature.

    Returns a dict with paradigm-specific metrics suitable for display.
    """
    metrics: dict[str, Any] = {}

    if sig.signature_type == "erp":
        for name, evk in sig.data["evokeds"].items():
            evk_uv = evk * 1e6  # V -> uV
            mean_evk = evk_uv.mean(axis=0)
            peak_idx = np.argmax(np.abs(mean_evk))
            metrics[f"{name}_peak_amplitude_uV"] = float(mean_evk[peak_idx])
            metrics[f"{name}_peak_latency_ms"] = float(
                sig.data["times"][peak_idx] * 1000
            )
            metrics[f"{name}_mean_amplitude_uV"] = float(np.mean(mean_evk))
            metrics[f"{name}_n_trials"] = sig.metadata["n_trials"].get(name, 0)
            metrics[f"{name}_rms_uV"] = float(np.sqrt(np.mean(mean_evk**2)))

    elif sig.signature_type == "erd_ers":
        for name, tfr in sig.data["tfr"].items():
            mean_tfr = tfr.mean(axis=0)
            metrics[f"{name}_max_erd_pct"] = float(np.min(mean_tfr))
            metrics[f"{name}_max_ers_pct"] = float(np.max(mean_tfr))
            freqs = sig.data["freqs"]
            erd_freq_idx = np.unravel_index(np.argmin(mean_tfr), mean_tfr.shape)[0]
            metrics[f"{name}_erd_peak_freq_hz"] = float(freqs[erd_freq_idx])
            metrics[f"{name}_n_trials"] = sig.metadata["n_trials"].get(name, 0)
            mu_mask = (freqs >= 8) & (freqs <= 13)
            if mu_mask.any():
                metrics[f"{name}_mu_band_erd_pct"] = float(mean_tfr[mu_mask, :].mean())
            beta_mask = (freqs >= 13) & (freqs <= 30)
            if beta_mask.any():
                metrics[f"{name}_beta_band_erd_pct"] = float(mean_tfr[beta_mask, :].mean())

    elif sig.signature_type == "psd_snr":
        for name in sig.data["event_names"]:
            if name in sig.data["snr"]:
                metrics[f"{name}_snr"] = float(sig.data["snr"][name])
            if name in sig.data["psd"]:
                psd = sig.data["psd"][name]
                metrics[f"{name}_peak_power"] = float(np.max(psd))
                peak_idx = np.argmax(psd)
                metrics[f"{name}_peak_freq_hz"] = float(sig.data["freqs"][peak_idx])
            metrics[f"{name}_n_trials"] = sig.metadata["n_trials"].get(name, 0)

    elif sig.signature_type == "cvep_response":
        for name, evk in sig.data["evokeds"].items():
            evk_uv = evk * 1e6
            mean_evk = evk_uv.mean(axis=0)
            metrics[f"{name}_peak_amplitude_uV"] = float(np.max(np.abs(mean_evk)))
            metrics[f"{name}_rms_uV"] = float(np.sqrt(np.mean(mean_evk**2)))
            metrics[f"{name}_n_trials"] = sig.metadata["n_trials"].get(name, 0)

    elif sig.signature_type == "rstate_psd":
        for name, bp in sig.data["band_powers"].items():
            for band, power in bp.items():
                metrics[f"{name}_{band}_pct"] = float(power)
            metrics[f"{name}_n_trials"] = sig.metadata["n_trials"].get(name, 0)
            if name in sig.data["psd"]:
                freqs = sig.data["freqs"]
                psd = sig.data["psd"][name]
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                if alpha_mask.any():
                    alpha_psd = psd[alpha_mask]
                    alpha_freqs = freqs[alpha_mask]
                    metrics[f"{name}_alpha_peak_hz"] = float(
                        alpha_freqs[np.argmax(alpha_psd)]
                    )

    return metrics


def _format_metric_value(key: str, val) -> str:
    """Format a single metric value with proper units."""
    if isinstance(val, float):
        if "hz" in key.lower() or "freq" in key.lower():
            return f"{val:.1f} Hz"
        if "latency" in key.lower():
            return f"{val:.0f} ms"
        if "uv" in key.lower() or "amplitude" in key.lower():
            return f"{val:.2f} \u00b5V"
        if "snr" in key.lower():
            return f"{val:.2f}"
        if "pct" in key or "erd" in key or "ers" in key:
            return f"{val:.1f}%"
        return f"{val:.4g}"
    return str(val)


def _build_metrics_table(metrics: dict[str, Any], sig_type: str) -> str:
    """Build a clean HTML metrics table for the branded HTML wrapper."""
    if not metrics:
        return ""

    # Group metrics by event/condition name (first token before _)
    groups: dict[str, list[tuple[str, str]]] = {}
    for k, v in metrics.items():
        parts = k.split("_", 1)
        group = parts[0] if len(parts) > 1 else "General"
        label = parts[1].replace("_", " ").title() if len(parts) > 1 else k
        groups.setdefault(group, []).append((label, _format_metric_value(k, v)))

    rows = []
    for group_name, items in groups.items():
        rows.append(
            f'<tr><td colspan="2" style="padding:8px 0 4px;font-weight:700;'
            f'color:{MOABB_NAVY};border-bottom:1px solid {MOABB_NAVY};'
            f'font-size:12px;letter-spacing:0.5px">{group_name}</td></tr>'
        )
        for label, val in items:
            rows.append(
                f'<tr><td style="padding:3px 12px 3px 0;color:{GRID_COLOR};'
                f'font-size:11px">{label}</td>'
                f'<td style="padding:3px 0;font-weight:600;'
                f'color:{MOABB_DARK_TEXT};font-size:11px;text-align:right">'
                f'{val}</td></tr>'
            )

    return (
        '<table style="border-collapse:collapse;font-family:Georgia,serif;'
        'width:100%">' + "".join(rows) + "</table>"
    )


def _wrap_branded_html(
    plotly_div: str,
    title: str,
    subtitle: str,
    metrics_html: str,
    ds_name: str,
) -> str:
    """Wrap a Plotly div in a branded MOABB HTML page."""
    source = _DEFAULT_SOURCE.format(version=moabb.__version__)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} \u2014 {ds_name} \u2014 MOABB</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');
  :root {{
    --navy: {MOABB_NAVY};
    --teal: {MOABB_TEAL};
    --sky: {MOABB_SKY};
    --coral: {MOABB_CORAL};
    --amber: {MOABB_AMBER};
    --dark-text: {MOABB_DARK_TEXT};
    --grid: {GRID_COLOR};
    --bg: #FAFBFC;
    --card-bg: #FFFFFF;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Source Serif 4', Georgia, Cambria, 'Times New Roman', serif;
    background: var(--bg);
    color: var(--dark-text);
    -webkit-font-smoothing: antialiased;
  }}
  .page {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 24px 48px;
  }}
  /* ---- Accent bar (Economist signature) ---- */
  .accent-bar {{
    height: 4px;
    background: var(--navy);
    margin-bottom: 0;
  }}
  .accent-pip {{
    width: 40px;
    height: 3px;
    background: var(--teal);
    margin-bottom: 24px;
  }}
  /* ---- Header ---- */
  .header {{
    padding: 32px 0 20px;
  }}
  .header h1 {{
    font-size: 26px;
    font-weight: 700;
    color: var(--dark-text);
    line-height: 1.2;
    letter-spacing: -0.3px;
    margin-bottom: 6px;
  }}
  .header .subtitle {{
    font-size: 15px;
    color: var(--grid);
    font-weight: 400;
    line-height: 1.4;
  }}
  /* ---- Main grid ---- */
  .content {{
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 24px;
    align-items: start;
  }}
  @media (max-width: 900px) {{
    .content {{ grid-template-columns: 1fr; }}
  }}
  /* ---- Chart card ---- */
  .chart-card {{
    background: var(--card-bg);
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    overflow: hidden;
  }}
  .chart-card .plotly-graph-div {{
    width: 100% !important;
  }}
  /* ---- Metrics sidebar ---- */
  .metrics-card {{
    background: var(--card-bg);
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    padding: 20px;
    position: sticky;
    top: 16px;
  }}
  .metrics-card h2 {{
    font-size: 13px;
    font-weight: 700;
    color: var(--navy);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--navy);
  }}
  .metrics-card h2::after {{
    content: '';
    display: block;
    width: 24px;
    height: 2px;
    background: var(--teal);
    margin-top: 2px;
  }}
  /* ---- Footer ---- */
  .source {{
    margin-top: 24px;
    padding-top: 12px;
    border-top: 1px solid #E0E4E8;
    font-size: 11px;
    color: var(--grid);
    font-style: italic;
  }}
</style>
</head>
<body>
<div class="page">
  <div class="accent-bar"></div>
  <div class="accent-pip"></div>
  <div class="header">
    <h1>{title}</h1>
    <div class="subtitle">{subtitle}</div>
  </div>
  <div class="content">
    <div class="chart-card">{plotly_div}</div>
    <div class="metrics-card">
      <h2>Objective Metrics</h2>
      {metrics_html}
    </div>
  </div>
  <div class="source">{source}</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_erp_interactive(
    sig: NeuralSignatureData,
    channel_idx: int = 0,
) -> "go.Figure":
    """Plot P300/ERP waveforms with confidence bands.

    Parameters
    ----------
    sig : NeuralSignatureData
        Output of :func:`compute_erp_signature`.
    channel_idx : int
        Channel index for the initial view.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    :func:`compute_erp_signature` : Compute the data for this plot.
    """
    import plotly.graph_objects as go

    _check_plotly()

    times_ms = sig.data["times"] * 1000
    event_names = sig.data["event_names"]
    ch_names = sig.metadata["ch_names"]
    n_trials = sig.metadata["n_trials"]

    fig = _make_figure()

    buttons = []
    active_events = [e for e in event_names if e in sig.data["evokeds"]]
    n_ev = len(active_events)
    traces_per_ch = n_ev * 3  # line + upper CI + lower CI

    for ch_i, ch_name in enumerate(ch_names):
        for ev_i, name in enumerate(active_events):
            evk = sig.data["evokeds"][name][ch_i] * 1e6
            sem = sig.data["sems"][name][ch_i] * 1e6
            color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]
            n_t = n_trials.get(name, "?")
            visible = ch_i == channel_idx

            # Main line
            fig.add_trace(go.Scatter(
                x=times_ms, y=evk, mode="lines",
                name=f"{name}  n={n_t}",
                line=dict(color=color, width=2.5),
                visible=visible,
                legendgroup=f"{name}_{ch_i}", showlegend=True,
            ))
            # Upper CI bound (invisible)
            fig.add_trace(go.Scatter(
                x=times_ms, y=evk + sem, mode="lines",
                line=dict(width=0), showlegend=False,
                visible=visible, legendgroup=f"{name}_{ch_i}",
                hoverinfo="skip",
            ))
            # Lower CI bound + fill
            fig.add_trace(go.Scatter(
                x=times_ms, y=evk - sem, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=_hex_to_rgba(color, 0.10),
                showlegend=False, visible=visible,
                legendgroup=f"{name}_{ch_i}", hoverinfo="skip",
            ))

    # Channel selector
    for ch_i, ch_name in enumerate(ch_names):
        vis = []
        for c_i in range(len(ch_names)):
            vis.extend([c_i == ch_i] * traces_per_ch)
        buttons.append(dict(label=ch_name, method="update", args=[{"visible": vis}]))

    if len(ch_names) > 1:
        fig.update_layout(updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True,
            x=1.0, xanchor="right", y=1.12, yanchor="top",
            bgcolor="white", bordercolor=MOABB_NAVY, borderwidth=1,
            font=dict(size=11, family=_FONT_FAMILY, color=MOABB_DARK_TEXT),
        )])

    # Stimulus onset line
    fig.add_vline(x=0, line_dash="dash", line_color=MOABB_NAVY, line_width=1,
                  opacity=0.4)
    fig.add_annotation(
        text="onset", x=2, y=1.0, xref="x", yref="paper",
        showarrow=False, font=dict(size=9, color=GRID_COLOR),
    )

    _add_branding(
        fig,
        title=f"Evoked Response \u2014 {sig.dataset_name}",
        subtitle=f"Grand average across {sum(n_trials.values())} trials "
                 f"\u00b7 Channel: {ch_names[channel_idx]}",
    )
    fig.update_layout(
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude (\u00b5V)",
        hovermode="x unified",
        height=480,
        margin=dict(t=100, b=60, l=72, r=100),
    )
    _add_head_inset(fig, ch_names, active_idx=channel_idx)

    return fig


def plot_erd_ers_interactive(
    sig: NeuralSignatureData,
) -> "go.Figure":
    """Plot ERD/ERS time-frequency heatmaps for motor imagery.

    Parameters
    ----------
    sig : NeuralSignatureData
        Output of :func:`compute_erd_ers_signature`.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    :func:`compute_erd_ers_signature` : Compute the data for this plot.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_plotly()

    event_names = [e for e in sig.data["event_names"] if e in sig.data["tfr"]]
    ch_names = sig.metadata["ch_names"]
    n_trials = sig.metadata["n_trials"]
    times = sig.data["times"]
    freqs = sig.data["freqs"]
    colorscale = get_plotly_colorscale()

    n_events = len(event_names)
    if n_events == 0:
        fig = _make_figure()
        fig.update_layout(title="No ERD/ERS data available")
        return fig

    total_trials = sum(n_trials.get(e, 0) for e in event_names)

    # Reserve right 18% of figure width for head inset + colorbar
    _heatmap_right = 0.78
    fig = make_subplots(
        rows=1, cols=n_events,
        subplot_titles=[
            f"<b>{name}</b>  <span style='color:{GRID_COLOR};font-size:11px'>"
            f"n={n_trials.get(name, '?')}</span>"
            for name in event_names
        ],
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[1] * n_events,
    )
    fig.update_layout(template=get_plotly_template())
    # Squeeze heatmap columns into left portion
    col_width = _heatmap_right / n_events
    gap = 0.02
    for i in range(1, n_events + 1):
        x0 = (i - 1) * col_width + gap
        x1 = i * col_width - gap
        fig.update_layout(**{
            f"xaxis{i if i > 1 else ''}": dict(domain=[x0, x1]),
        })

    # Global symmetric color range from the 95th percentile
    all_values = [sig.data["tfr"][n] for n in event_names]
    vmax = max(np.percentile(np.abs(v), 95) for v in all_values) if all_values else 100

    # Channel slider data
    steps = []
    for ch_i, ch_name in enumerate(ch_names):
        step_data = [sig.data["tfr"][name][ch_i] for name in event_names]
        steps.append(dict(method="update", label=ch_name, args=[{"z": step_data}]))

    # Initial heatmaps (first channel)
    for ev_i, name in enumerate(event_names):
        tfr = sig.data["tfr"][name][0]
        fig.add_trace(
            go.Heatmap(
                z=tfr, x=times, y=freqs,
                colorscale=colorscale, zmin=-vmax, zmax=vmax,
                colorbar=dict(
                    title=dict(text="ERD/ERS (%)", font=dict(size=11)),
                    thickness=12, len=0.3,
                    y=0.95, yanchor="top",
                    x=0.90, xanchor="center",
                    tickfont=dict(size=10),
                    outlinewidth=0,
                ) if ev_i == n_events - 1 else None,
                showscale=(ev_i == n_events - 1),
                hovertemplate=(
                    "<b>%{y:.0f} Hz</b> at %{x:.2f}s"
                    "<br>ERD/ERS: %{z:.1f}%<extra></extra>"
                ),
            ),
            row=1, col=ev_i + 1,
        )

    # Add head inset in the right-side space (vertically centered with heatmaps)
    n_head_traces = _add_head_inset(
        fig, ch_names, active_idx=0,
        xdomain=(0.83, 0.99), ydomain=(0.10, 0.60),
    )
    # n_events heatmap traces + 1 "all dots" trace + n_head_traces highlight traces
    # The highlight traces start at index (n_events + 1)
    head_start = n_events + 1  # skip heatmaps + all-dots trace

    # Channel slider — update heatmap z-data AND head highlight visibility
    if len(ch_names) > 1:
        for ch_i, step in enumerate(steps):
            vis = [True] * n_events  # heatmaps always visible
            vis.append(True)  # all-dots trace always visible
            vis.extend(i == ch_i for i in range(n_head_traces))
            step["args"] = [{"z": step["args"][0]["z"], "visible": vis}]

        fig.update_layout(sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="Channel: ", font=dict(size=12, color=MOABB_DARK_TEXT),
            ),
            pad=dict(t=40),
            steps=steps,
            bordercolor=MOABB_NAVY, borderwidth=1,
            activebgcolor=_hex_to_rgba(MOABB_TEAL, 0.25),
            font=dict(size=11),
        )])

    # Mu/beta band reference lines
    for ev_i in range(n_events):
        for band_freq in (8, 13):
            fig.add_hline(
                y=band_freq, line_dash="dot",
                line_color="rgba(47, 62, 92, 0.25)", line_width=0.8,
                row=1, col=ev_i + 1,
            )

    # Stimulus onset
    for ev_i in range(n_events):
        fig.add_vline(
            x=0, line_dash="dash",
            line_color="rgba(47, 62, 92, 0.35)", line_width=0.8,
            row=1, col=ev_i + 1,
        )

    _add_branding(
        fig,
        title=f"ERD/ERS Time-Frequency \u2014 {sig.dataset_name}",
        subtitle=(
            f"Grand average \u00b7 {total_trials} trials "
            f"\u00b7 Channels: {', '.join(ch_names)} "
            f"\u00b7 Baseline-corrected percentage change"
        ),
    )
    fig.update_layout(
        height=520, margin=dict(t=110, b=80, l=72, r=20),
    )
    for i in range(1, n_events + 1):
        fig.update_xaxes(title_text="Time (s)", row=1, col=i)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)

    return fig


def plot_ssvep_interactive(
    sig: NeuralSignatureData,
) -> "go.Figure":
    """Plot SSVEP power spectra with stimulus frequency markers.

    Parameters
    ----------
    sig : NeuralSignatureData
        Output of :func:`compute_ssvep_signature`.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    :func:`compute_ssvep_signature` : Compute the data for this plot.
    """
    import plotly.graph_objects as go

    _check_plotly()

    freqs = sig.data["freqs"]
    stim_freqs = sig.data["stimulus_frequencies"]
    event_names = sig.data["event_names"]
    n_trials = sig.metadata["n_trials"]
    fig = _make_figure()

    for ev_i, name in enumerate(event_names):
        if name not in sig.data["psd"]:
            continue
        psd = sig.data["psd"][name]
        n_t = n_trials.get(name, "?")
        snr_val = sig.data["snr"].get(name, None)
        color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]

        label = f"{name} Hz  n={n_t}"
        if snr_val is not None and snr_val > 0:
            label += f"  SNR {snr_val:.1f}"

        fig.add_trace(go.Scatter(
            x=freqs, y=psd, mode="lines", name=label,
            line=dict(color=color, width=2.5),
        ))

    # Stimulus frequency markers
    for freq in stim_freqs:
        for harmonic in [1, 2]:
            f = freq * harmonic
            if freqs is not None and f <= freqs[-1]:
                fig.add_vline(
                    x=f,
                    line_dash="dash" if harmonic == 1 else "dot",
                    line_color=_hex_to_rgba(MOABB_NAVY, 0.3),
                    line_width=1,
                )
                if harmonic == 1:
                    fig.add_annotation(
                        x=f, y=1.06, xref="x", yref="paper",
                        text=f"<b>{f:.0f} Hz</b>", showarrow=False,
                        font=dict(size=9, color=MOABB_NAVY),
                    )

    total_trials = sum(n_trials.values())
    _add_branding(
        fig,
        title=f"SSVEP Power Spectrum \u2014 {sig.dataset_name}",
        subtitle=f"Grand average \u00b7 {total_trials} trials \u00b7 Welch PSD",
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density (V\u00b2/Hz)",
        yaxis_type="log",
        hovermode="x unified",
        height=480,
        margin=dict(t=100, b=60, l=72, r=24),
    )

    return fig


def plot_cvep_interactive(
    sig: NeuralSignatureData,
    channel_idx: int = 0,
) -> "go.Figure":
    """Plot c-VEP evoked responses and PSD.

    Parameters
    ----------
    sig : NeuralSignatureData
        Output of :func:`compute_cvep_signature`.
    channel_idx : int
        Channel index for evoked response view.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    :func:`compute_cvep_signature` : Compute the data for this plot.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_plotly()

    event_names = [e for e in sig.data["event_names"] if e in sig.data["evokeds"]]
    ch_names = sig.metadata["ch_names"]
    n_trials = sig.metadata["n_trials"]
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "<b>Evoked Response</b>",
            "<b>Power Spectrum</b>",
        ],
        vertical_spacing=0.18,
        row_heights=[0.6, 0.4],
    )
    fig.update_layout(template=get_plotly_template())

    times_ms = sig.data["times"] * 1000
    for ev_i, name in enumerate(event_names):
        color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]
        n_t = n_trials.get(name, "?")
        ch_i = min(channel_idx, sig.data["evokeds"][name].shape[0] - 1)

        fig.add_trace(go.Scatter(
            x=times_ms, y=sig.data["evokeds"][name][ch_i] * 1e6,
            mode="lines", name=f"Code {name}  n={n_t}",
            line=dict(color=color, width=2.5), legendgroup=name,
        ), row=1, col=1)

        if name in sig.data["psd"] and sig.data["freqs"] is not None:
            fig.add_trace(go.Scatter(
                x=sig.data["freqs"], y=sig.data["psd"][name],
                mode="lines", name=f"PSD {name}",
                line=dict(color=color, width=2, dash="dot"),
                legendgroup=name, showlegend=False,
            ), row=2, col=1)

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (\u00b5V)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="PSD (V\u00b2/Hz)", type="log", row=2, col=1)

    ch_label = ch_names[channel_idx] if channel_idx < len(ch_names) else ""
    _add_branding(
        fig,
        title=f"c-VEP Signature \u2014 {sig.dataset_name}",
        subtitle=f"Channel: {ch_label} \u00b7 {sum(n_trials.values())} trials",
    )
    fig.update_layout(
        height=640, hovermode="x unified",
        margin=dict(t=110, b=60, l=72, r=100),
    )
    _add_head_inset(fig, ch_names, active_idx=channel_idx)
    return fig


def plot_rstate_interactive(
    sig: NeuralSignatureData,
) -> "go.Figure":
    """Plot resting state PSD overlay and band power bar chart.

    Parameters
    ----------
    sig : NeuralSignatureData
        Output of :func:`compute_rstate_signature`.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    :func:`compute_rstate_signature` : Compute the data for this plot.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _check_plotly()

    event_names = [e for e in sig.data["event_names"] if e in sig.data["psd"]]
    n_trials = sig.metadata["n_trials"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Power Spectral Density</b>",
            "<b>Relative Band Power</b>",
        ],
        horizontal_spacing=0.14,
        column_widths=[0.6, 0.4],
    )
    fig.update_layout(template=get_plotly_template())

    freqs = sig.data["freqs"]
    band_names = list(sig.data["bands"].keys())
    band_labels = [b.capitalize() for b in band_names]

    for ev_i, name in enumerate(event_names):
        color = _PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)]
        n_t = n_trials.get(name, "?")

        fig.add_trace(go.Scatter(
            x=freqs, y=sig.data["psd"][name],
            mode="lines", name=f"{name}  n={n_t}",
            line=dict(color=color, width=2.5), legendgroup=name,
        ), row=1, col=1)

        if name in sig.data["band_powers"]:
            bp = sig.data["band_powers"][name]
            fig.add_trace(go.Bar(
                x=band_labels, y=[bp[b] for b in band_names],
                name=name, marker_color=color,
                marker_line_width=0,
                legendgroup=name, showlegend=False,
            ), row=1, col=2)

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="PSD (V\u00b2/Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Relative Power (%)", row=1, col=2)

    total_trials = sum(n_trials.values())
    _add_branding(
        fig,
        title=f"Resting State Signature \u2014 {sig.dataset_name}",
        subtitle=f"Grand average \u00b7 {total_trials} trials \u00b7 Welch PSD",
    )
    fig.update_layout(
        height=480, hovermode="x unified", barmode="group",
        margin=dict(t=110, b=60, l=72, r=24),
    )
    return fig


# ---------------------------------------------------------------------------
# Paradigm dispatch
# ---------------------------------------------------------------------------

_PARADIGM_HANDLERS = {
    "imagery": (compute_erd_ers_signature, plot_erd_ers_interactive),
    "p300": (compute_erp_signature, plot_erp_interactive),
    "ssvep": (compute_ssvep_signature, plot_ssvep_interactive),
    "cvep": (compute_cvep_signature, plot_cvep_interactive),
    "rstate": (compute_rstate_signature, plot_rstate_interactive),
}


def _get_paradigm_instance(paradigm_name: str, dataset=None):
    """Instantiate the appropriate paradigm for data loading."""
    from moabb.paradigms import (
        CVEP,
        SSVEP,
        MotorImagery,
        P300,
        RestingStateToP300Adapter,
    )

    _paradigm_factories = {
        "imagery": lambda _ds: MotorImagery(n_classes=2, fmin=1, fmax=45, resample=128),
        "p300": lambda _ds: P300(),
        "ssvep": lambda _ds: SSVEP(n_classes=2, resample=256),
        "cvep": lambda _ds: CVEP(n_classes=2, resample=256),
        "rstate": lambda ds: RestingStateToP300Adapter(
            events=(
                list(ds.event_id.keys())
                if ds is not None and hasattr(ds, "event_id")
                else None
            )
        ),
    }
    factory = _paradigm_factories.get(paradigm_name)
    if factory is None:
        raise ValueError(
            f"Unknown paradigm {paradigm_name!r}. "
            f"Supported: {list(_paradigm_factories)}"
        )
    return factory(dataset)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def _load_and_compute(
    dataset,
    subjects: list[int],
    compute_fn,
    plot_fn,
    *,
    collect_per_subject: bool = False,
) -> tuple["NeuralSignatureData | None", "go.Figure | None", dict]:
    """Load epochs, compute grand-average signature, and plot.

    Returns ``(sig, fig, per_subject_epochs)`` where *per_subject_epochs*
    is non-empty only when *collect_per_subject* is True.
    """
    paradigm_name = dataset.paradigm
    paradigm = _get_paradigm_instance(paradigm_name, dataset)
    ds_name = dataset.__class__.__name__
    code = dataset.code

    # Widen the dataset interval to include a pre-stimulus baseline
    _BASELINE_MARGIN = 0.5  # seconds before the task interval
    _task_onset = None
    if hasattr(dataset, "interval") and dataset.interval is not None:
        orig = dataset.interval
        _task_onset = orig[0]
        dataset.interval = [orig[0] - _BASELINE_MARGIN, orig[1]]

    per_subject_epochs: dict = {}
    grand_epochs = None

    for subj in subjects:
        log.info("Loading subject %s for %s", subj, ds_name)
        try:
            epochs, _, _ = paradigm.get_data(
                dataset, [subj], return_epochs=True
            )
        except Exception as exc:
            log.warning("Failed to load subject %s: %s", subj, exc)
            continue
        if collect_per_subject:
            per_subject_epochs[subj] = epochs
        if grand_epochs is None:
            grand_epochs = epochs
        else:
            try:
                grand_epochs = mne.concatenate_epochs(
                    [grand_epochs, epochs], verbose=False
                )
            except Exception:
                log.warning("Cannot concatenate epochs for subject %s", subj)

    if grand_epochs is None or len(grand_epochs) == 0:
        return None, None, per_subject_epochs

    log.info("Computing grand-average signature for %s", ds_name)
    sig = compute_fn(grand_epochs)
    sig.dataset_name = ds_name
    sig.dataset_code = code
    _relabel_signature(sig, _build_event_label_map(dataset))

    fig = plot_fn(sig)
    return sig, fig, per_subject_epochs


def generate_neural_signature(
    dataset,
    subjects: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate interactive neural signature HTML files for a dataset.

    Produces branded, standalone HTML files with the MOABB editorial style.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    subjects : list of int or None
        Subject IDs.  ``None`` uses all subjects.
    output_dir : str, Path, or None
        Output directory.  Defaults to ``./neural_signatures/``.

    Returns
    -------
    list of Path
        Paths to the generated HTML files.
    """
    _check_plotly()

    if output_dir is None:
        output_dir = Path("neural_signatures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = dataset.subject_list

    paradigm_name = dataset.paradigm
    if paradigm_name not in _PARADIGM_HANDLERS:
        raise ValueError(
            f"Unsupported paradigm {paradigm_name!r} for dataset "
            f"{dataset.__class__.__name__}. "
            f"Supported: {list(_PARADIGM_HANDLERS)}"
        )

    compute_fn, plot_fn = _PARADIGM_HANDLERS[paradigm_name]
    code = dataset.code
    ds_name = dataset.__class__.__name__
    generated = []

    sig, fig, all_evokeds_per_subject = _load_and_compute(
        dataset, subjects, compute_fn, plot_fn, collect_per_subject=True,
    )
    if sig is None:
        log.warning("No valid epochs for %s, skipping.", ds_name)
        return generated

    metrics = compute_metrics(sig)
    metrics_html = _build_metrics_table(metrics, sig.signature_type)

    # Title / subtitle for the HTML wrapper
    _sig_titles = {
        "erd_ers": "ERD/ERS Time-Frequency",
        "erp": "Evoked Response",
        "psd_snr": "SSVEP Power Spectrum",
        "cvep_response": "c-VEP Signature",
        "rstate_psd": "Resting State",
    }
    title = f"{_sig_titles.get(sig.signature_type, 'Neural Signature')} \u2014 {ds_name}"
    n_total = sum(sig.metadata.get("n_trials", {}).values())
    subtitle = (
        f"Grand average across {len(subjects)} subject(s), "
        f"{n_total} total trials"
    )

    # Write branded HTML
    plotly_div = fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"displayModeBar": True, "displaylogo": False},
    )
    html = _wrap_branded_html(plotly_div, title, subtitle, metrics_html, ds_name)
    path = output_dir / f"{code}_grand_average.html"
    path.write_text(html, encoding="utf-8")
    generated.append(path)
    log.info("Wrote %s", path)

    # ---- Per-channel ----
    if paradigm_name in ("p300", "cvep"):
        fig_ch = _make_per_channel_figure(sig, paradigm_name)
        if fig_ch is not None:
            path_ch = output_dir / f"{code}_per_channel.html"
            fig_ch.write_html(str(path_ch), include_plotlyjs=True)
            generated.append(path_ch)
            log.info("Wrote %s", path_ch)

    # ---- Per-subject ----
    if len(all_evokeds_per_subject) > 1:
        fig_subj = _make_per_subject_figure(
            all_evokeds_per_subject, compute_fn, plot_fn,
            paradigm_name, ds_name, code,
        )
        if fig_subj is not None:
            path_subj = output_dir / f"{code}_per_subject.html"
            fig_subj.write_html(str(path_subj), include_plotlyjs=True)
            generated.append(path_subj)
            log.info("Wrote %s", path_subj)

    return generated


def _make_per_channel_figure(sig, paradigm_name):
    """Create a per-channel overview figure for ERP-type signatures."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ch_names = sig.metadata["ch_names"]
    if len(ch_names) <= 1:
        return None

    n_ch = min(len(ch_names), 16)
    ncols = min(4, n_ch)
    nrows = (n_ch + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"<b>{ch}</b>" for ch in ch_names[:n_ch]],
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.04,
    )
    fig.update_layout(template=get_plotly_template())

    times_ms = sig.data["times"] * 1000
    event_names = sig.data.get("event_names", [])

    for ch_i in range(n_ch):
        row = ch_i // ncols + 1
        col = ch_i % ncols + 1
        for ev_i, name in enumerate(event_names):
            if name not in sig.data.get("evokeds", {}):
                continue
            evk = sig.data["evokeds"][name]
            if ch_i >= evk.shape[0]:
                continue
            fig.add_trace(go.Scatter(
                x=times_ms, y=evk[ch_i] * 1e6, mode="lines",
                name=name, line=dict(color=_PLOT_PALETTE[ev_i % len(_PLOT_PALETTE)], width=1.5),
                showlegend=(ch_i == 0), legendgroup=name,
            ), row=row, col=col)

    _add_branding(fig, title=f"Per-Channel ERP \u2014 {sig.dataset_name}", subtitle="")
    fig.update_layout(height=220 * nrows + 80, hovermode="x unified")
    return fig


def _make_per_subject_figure(
    subjects_epochs, compute_fn, plot_fn, paradigm_name, ds_name, code,
):
    """Create a per-subject overlay figure."""
    import plotly.graph_objects as go

    _check_plotly()
    fig = _make_figure()

    for i, (subj, epochs) in enumerate(subjects_epochs.items()):
        try:
            sig = compute_fn(epochs)
            sig.dataset_name = ds_name
            sig.dataset_code = code
        except Exception:
            continue

        color = _PLOT_PALETTE[i % len(_PLOT_PALETTE)]

        if paradigm_name == "imagery":
            event_names = [
                e for e in sig.data.get("event_names", [])
                if e in sig.data.get("tfr", {})
            ]
            if not event_names:
                continue
            # Show mu-band (8-13 Hz) mean power over time per subject
            freqs = sig.data["freqs"]
            mu_mask = (freqs >= 8) & (freqs <= 13)
            tfr = sig.data["tfr"][event_names[0]]  # (ch, freq, time)
            mu_power = tfr[:, mu_mask, :].mean(axis=(0, 1))
            fig.add_trace(go.Scatter(
                x=sig.data["times"], y=mu_power,
                mode="lines", name=f"Sub {subj}",
                line=dict(color=color, width=1.5), opacity=0.7,
            ))
        elif paradigm_name in ("p300", "cvep"):
            event_names = [
                e for e in sig.data.get("event_names", [])
                if e in sig.data.get("evokeds", {})
            ]
            if not event_names:
                continue
            evk = sig.data["evokeds"][event_names[0]]
            fig.add_trace(go.Scatter(
                x=sig.data["times"] * 1000, y=evk.mean(axis=0) * 1e6,
                mode="lines", name=f"Sub {subj}",
                line=dict(color=color, width=1.5), opacity=0.7,
            ))
        elif paradigm_name in ("ssvep", "rstate"):
            event_names = [
                e for e in sig.data.get("event_names", [])
                if e in sig.data.get("psd", {})
            ]
            if not event_names:
                continue
            fig.add_trace(go.Scatter(
                x=sig.data["freqs"], y=sig.data["psd"][event_names[0]],
                mode="lines", name=f"Sub {subj}",
                line=dict(color=color, width=1.5), opacity=0.7,
            ))

    _add_branding(
        fig,
        title=f"Per-Subject Overview \u2014 {ds_name}",
        subtitle=f"{len(subjects_epochs)} subjects",
    )
    fig.update_layout(hovermode="x unified")

    if paradigm_name == "imagery":
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Mu-band (8\u201313 Hz) Power",
        )
    elif paradigm_name in ("p300", "cvep"):
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Amplitude (\u00b5V)")
        fig.add_vline(x=0, line_dash="dash", line_color=MOABB_NAVY,
                      line_width=1, opacity=0.4)
    elif paradigm_name in ("ssvep", "rstate"):
        fig.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="PSD (V\u00b2/Hz)",
            yaxis_type="log",
        )
    return fig


def neural_signature_html(
    dataset,
    subjects: list[int] | None = None,
) -> dict[str, str]:
    """Generate neural signature figures and return HTML strings.

    Parameters
    ----------
    dataset : BaseDataset
        A MOABB dataset instance.
    subjects : list of int or None
        Subject IDs.

    Returns
    -------
    dict of str -> str
        Mapping of figure name to HTML string.
    """
    _check_plotly()

    if subjects is None:
        subjects = dataset.subject_list

    paradigm_name = dataset.paradigm
    if paradigm_name not in _PARADIGM_HANDLERS:
        raise ValueError(f"Unsupported paradigm {paradigm_name!r}")

    compute_fn, plot_fn = _PARADIGM_HANDLERS[paradigm_name]

    sig, fig, _ = _load_and_compute(dataset, subjects, compute_fn, plot_fn)
    if sig is None:
        return {}

    return {"grand_average": fig.to_html(include_plotlyjs=True)}
