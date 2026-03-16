"""Tests for moabb.analysis.neural_signatures."""

import tempfile
from pathlib import Path

import pytest

from moabb.datasets.fake import FakeDataset

# Skip entire module if plotly is not installed
plotly = pytest.importorskip("plotly")

from moabb.analysis.neural_signatures import (  # noqa: E402
    NeuralSignatureData,
    compute_cvep_signature,
    compute_erd_ers_signature,
    compute_erp_signature,
    compute_rstate_signature,
    compute_ssvep_signature,
    generate_neural_signature,
    get_plotly_colorscale,
    get_plotly_template,
    plot_cvep_interactive,
    plot_erd_ers_interactive,
    plot_erp_interactive,
    plot_rstate_interactive,
    plot_ssvep_interactive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def p300_dataset():
    return FakeDataset(
        paradigm="p300",
        event_list=("Target", "NonTarget"),
        n_subjects=2,
        n_sessions=1,
        n_runs=1,
        channels=("C3", "Cz", "C4", "Fz", "Pz", "Oz"),
        sfreq=128,
        n_events=40,
        duration=60,
    )


@pytest.fixture
def imagery_dataset():
    return FakeDataset(
        paradigm="imagery",
        event_list=("left_hand", "right_hand"),
        n_subjects=2,
        n_sessions=1,
        n_runs=1,
        channels=("C3", "Cz", "C4"),
        sfreq=128,
        n_events=30,
        duration=60,
    )


@pytest.fixture
def ssvep_dataset():
    return FakeDataset(
        paradigm="ssvep",
        event_list=("13.0", "15.0"),
        n_subjects=2,
        n_sessions=1,
        n_runs=1,
        channels=("O1", "Oz", "O2"),
        sfreq=256,
        n_events=30,
        duration=60,
    )


@pytest.fixture
def cvep_dataset():
    return FakeDataset(
        paradigm="cvep",
        event_list=("1.0", "0.0"),
        n_subjects=2,
        n_sessions=1,
        n_runs=1,
        channels=("O1", "Oz", "O2"),
        sfreq=256,
        n_events=30,
        duration=60,
    )


@pytest.fixture
def rstate_dataset():
    return FakeDataset(
        paradigm="rstate",
        event_list=("eyes_open", "eyes_closed"),
        n_subjects=2,
        n_sessions=1,
        n_runs=1,
        channels=("C3", "Cz", "C4", "Fz", "Pz", "Oz"),
        sfreq=128,
        n_events=20,
        duration=120,
    )


def _get_epochs(dataset, paradigm_cls, **kwargs):
    """Helper to load epochs from a FakeDataset."""
    paradigm = paradigm_cls(**kwargs)
    epochs, labels, meta = paradigm.get_data(
        dataset, subjects=[1], return_epochs=True
    )
    return epochs


# ---------------------------------------------------------------------------
# Style tests
# ---------------------------------------------------------------------------


class TestPlotlyStyle:
    def test_get_plotly_template(self):
        template = get_plotly_template()
        assert template is not None
        assert template.layout is not None
        assert len(template.layout.colorway) > 0

    def test_get_plotly_colorscale(self):
        cs = get_plotly_colorscale()
        assert len(cs) == 5
        assert cs[0][0] == 0.0
        assert cs[-1][0] == 1.0


# ---------------------------------------------------------------------------
# Data container tests
# ---------------------------------------------------------------------------


class TestNeuralSignatureData:
    def test_creation(self):
        sig = NeuralSignatureData(
            paradigm="p300",
            dataset_name="Test",
            dataset_code="test",
            signature_type="erp",
        )
        assert sig.paradigm == "p300"
        assert isinstance(sig.data, dict)
        assert isinstance(sig.metadata, dict)


# ---------------------------------------------------------------------------
# Computation tests
# ---------------------------------------------------------------------------


class TestComputeERP:
    def test_basic(self, p300_dataset):
        from moabb.paradigms import P300

        epochs = _get_epochs(p300_dataset, P300)
        sig = compute_erp_signature(epochs)

        assert sig.paradigm == "p300"
        assert sig.signature_type == "erp"
        # Event names may be "Target"/"NonTarget" or integer strings
        assert len(sig.data["evokeds"]) >= 2
        first_name = list(sig.data["evokeds"].keys())[0]
        assert sig.data["evokeds"][first_name].shape[0] == len(epochs.ch_names)
        assert len(sig.data["times"]) > 0
        first_trials = list(sig.metadata["n_trials"].values())[0]
        assert first_trials > 0

    def test_sem_shape(self, p300_dataset):
        from moabb.paradigms import P300

        epochs = _get_epochs(p300_dataset, P300)
        sig = compute_erp_signature(epochs)

        for name in sig.data["event_names"]:
            if name in sig.data["evokeds"]:
                assert sig.data["sems"][name].shape == sig.data["evokeds"][name].shape


class TestComputeERDERS:
    def test_basic(self, imagery_dataset):
        from moabb.paradigms import MotorImagery

        epochs = _get_epochs(imagery_dataset, MotorImagery, n_classes=2)
        sig = compute_erd_ers_signature(epochs)

        assert sig.paradigm == "imagery"
        assert sig.signature_type == "erd_ers"
        assert len(sig.data["tfr"]) > 0
        assert len(sig.data["freqs"]) > 0

    def test_channel_selection(self, imagery_dataset):
        from moabb.paradigms import MotorImagery

        epochs = _get_epochs(imagery_dataset, MotorImagery, n_classes=2)
        sig = compute_erd_ers_signature(epochs)
        # Should pick central channels
        for ch in sig.metadata["ch_names"]:
            assert ch in epochs.ch_names


class TestComputeSSVEP:
    def test_basic(self, ssvep_dataset):
        from moabb.paradigms import SSVEP

        epochs = _get_epochs(ssvep_dataset, SSVEP, n_classes=2)
        sig = compute_ssvep_signature(epochs)

        assert sig.paradigm == "ssvep"
        assert sig.signature_type == "psd_snr"
        assert len(sig.data["psd"]) > 0
        assert sig.data["freqs"] is not None

    def test_stimulus_frequencies(self, ssvep_dataset):
        from moabb.paradigms import SSVEP

        epochs = _get_epochs(ssvep_dataset, SSVEP, n_classes=2)
        sig = compute_ssvep_signature(epochs)

        assert len(sig.data["stimulus_frequencies"]) > 0


class TestComputeCVEP:
    def test_basic(self, cvep_dataset):
        from moabb.paradigms import CVEP

        epochs = _get_epochs(cvep_dataset, CVEP, n_classes=2)
        sig = compute_cvep_signature(epochs)

        assert sig.paradigm == "cvep"
        assert sig.signature_type == "cvep_response"
        assert len(sig.data["evokeds"]) > 0
        assert len(sig.data["psd"]) > 0


class TestComputeRState:
    def test_basic(self, rstate_dataset):
        from moabb.paradigms import RestingStateToP300Adapter

        events = list(rstate_dataset.event_id.keys())
        epochs = _get_epochs(
            rstate_dataset, RestingStateToP300Adapter,
            tmin=0, tmax=3, resample=128, events=events,
        )
        sig = compute_rstate_signature(epochs)

        assert sig.paradigm == "rstate"
        assert sig.signature_type == "rstate_psd"
        assert len(sig.data["psd"]) > 0
        assert len(sig.data["band_powers"]) > 0

    def test_band_powers_sum(self, rstate_dataset):
        from moabb.paradigms import RestingStateToP300Adapter

        events = list(rstate_dataset.event_id.keys())
        epochs = _get_epochs(
            rstate_dataset, RestingStateToP300Adapter,
            tmin=0, tmax=3, resample=128, events=events,
        )
        sig = compute_rstate_signature(epochs)

        for name, bp in sig.data["band_powers"].items():
            total = sum(bp.values())
            # Should be close to 100% (relative powers)
            assert 90 < total < 110, f"Band powers for {name} sum to {total}"


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------


class TestPlotERP:
    def test_produces_figure(self, p300_dataset):
        from moabb.paradigms import P300

        epochs = _get_epochs(p300_dataset, P300)
        sig = compute_erp_signature(epochs)
        sig.dataset_name = "Test"

        fig = plot_erp_interactive(sig)
        assert fig is not None
        assert len(fig.data) > 0

    def test_html_output(self, p300_dataset):
        from moabb.paradigms import P300

        epochs = _get_epochs(p300_dataset, P300)
        sig = compute_erp_signature(epochs)
        sig.dataset_name = "Test"

        fig = plot_erp_interactive(sig)
        html = fig.to_html(include_plotlyjs=True)
        assert "plotly" in html.lower()
        assert len(html) > 100


class TestPlotERDERS:
    def test_produces_figure(self, imagery_dataset):
        from moabb.paradigms import MotorImagery

        epochs = _get_epochs(imagery_dataset, MotorImagery, n_classes=2)
        sig = compute_erd_ers_signature(epochs)
        sig.dataset_name = "Test"

        fig = plot_erd_ers_interactive(sig)
        assert fig is not None
        assert len(fig.data) > 0


class TestPlotSSVEP:
    def test_produces_figure(self, ssvep_dataset):
        from moabb.paradigms import SSVEP

        epochs = _get_epochs(ssvep_dataset, SSVEP, n_classes=2)
        sig = compute_ssvep_signature(epochs)
        sig.dataset_name = "Test"

        fig = plot_ssvep_interactive(sig)
        assert fig is not None
        assert len(fig.data) > 0


class TestPlotCVEP:
    def test_produces_figure(self, cvep_dataset):
        from moabb.paradigms import CVEP

        epochs = _get_epochs(cvep_dataset, CVEP, n_classes=2)
        sig = compute_cvep_signature(epochs)
        sig.dataset_name = "Test"

        fig = plot_cvep_interactive(sig)
        assert fig is not None


class TestPlotRState:
    def test_produces_figure(self, rstate_dataset):
        from moabb.paradigms import RestingStateToP300Adapter

        events = list(rstate_dataset.event_id.keys())
        epochs = _get_epochs(
            rstate_dataset, RestingStateToP300Adapter,
            tmin=0, tmax=3, resample=128, events=events,
        )
        sig = compute_rstate_signature(epochs)
        sig.dataset_name = "Test"

        fig = plot_rstate_interactive(sig)
        assert fig is not None
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


class TestGenerateNeuralSignature:
    def test_p300_end_to_end(self, p300_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_neural_signature(
                p300_dataset, subjects=[1, 2], output_dir=tmpdir
            )
            assert len(paths) >= 1
            for p in paths:
                assert p.exists()
                content = p.read_text()
                assert "plotly" in content.lower()

    def test_imagery_end_to_end(self, imagery_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_neural_signature(
                imagery_dataset, subjects=[1], output_dir=tmpdir
            )
            assert len(paths) >= 1

    def test_ssvep_end_to_end(self, ssvep_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_neural_signature(
                ssvep_dataset, subjects=[1], output_dir=tmpdir
            )
            assert len(paths) >= 1

    def test_output_dir_created(self, p300_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "dir"
            paths = generate_neural_signature(
                p300_dataset, subjects=[1], output_dir=out
            )
            assert out.exists()

    def test_unsupported_paradigm_raises(self):
        ds = FakeDataset(paradigm="imagery")
        # Monkey-patch to an unsupported paradigm
        ds._paradigm = "unknown"
        object.__setattr__(ds, "paradigm", "unknown")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported paradigm"):
                generate_neural_signature(ds, subjects=[1], output_dir=tmpdir)


# ---------------------------------------------------------------------------
# Paradigm dispatch tests
# ---------------------------------------------------------------------------


class TestParadigmDispatch:
    def test_all_paradigms_have_handlers(self):
        from moabb.analysis.neural_signatures import _PARADIGM_HANDLERS

        expected = {"imagery", "p300", "ssvep", "cvep", "rstate"}
        assert set(_PARADIGM_HANDLERS.keys()) == expected

    def test_handler_tuple_structure(self):
        from moabb.analysis.neural_signatures import _PARADIGM_HANDLERS

        for name, handler in _PARADIGM_HANDLERS.items():
            assert len(handler) == 2, f"Handler for {name} should be 2-tuple"
            assert callable(handler[0]), f"compute_fn for {name} not callable"
            assert callable(handler[1]), f"plot_fn for {name} not callable"
