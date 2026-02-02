"""Alex Motor imagery dataset."""

from mne.io import Raw

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)

from . import download as dl
from .base import BaseDataset


ALEX_URL = "https://zenodo.org/record/806023/files/"


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset.

    Motor imagery dataset from the PhD dissertation of A. Barachant [1]_.

    This Dataset contains EEG recordings from 8 subjects, performing 2 task of
    motor imagination (right hand, feet or rest). Data have been recorded at
    512Hz with 16 wet electrodes (Fpz, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
    P7, P3, Pz, P4, P8) with a g.tec g.USBamp EEG amplifier.

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the motor imagination. The start of a trial is encoded as 1,
    then the actual start of the motor imagination is encoded with 2 for
    imagination of a right hand movement, 3 for imagination of both feet
    movement and 4 with a rest trial.

    The duration of each trial is 3 second. There is 20 trial of each class.

    references
    ----------
    .. [1] Barachant, A., 2012. Commande robuste d'un effecteur par une
           interface cerveau machine EEG asynchrone (Doctoral dissertation,
           Université de Grenoble).
           https://tel.archives-ouvertes.fr/tel-01196752
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=17,
            channel_types={"eeg": 16, "stim": 1},
            sensors=[
                "Fpz",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
            ],
            hardware="g.tec g.USBamp",
            sensor_type="wet",
            montage="standard_1020",
            line_freq=50.0,
            filters="0.1-100 Hz bandpass, 50 Hz notch",
            ground="AFz",
            reference="car",
            software="OpenViBE.",
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="healthy",
            clinical_population="spinal_cord_injury",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="right_hand_feet_rest",
            n_classes=3,
            trial_duration=3.0,
            tasks=["rest"],
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.806023",
            description="Motor imagery dataset from A. Barachant PhD dissertation",
            investigators=["Barachant, A."],
            institution="University of Grenoble",
            country="FR",
            repository="Zenodo",
            data_url="https://zenodo.org/record/806023",
            license="CC BY 4.0",
            publication_year=2012,
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["bci"]),
        data_processed=False,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events=dict(right_hand=2, feet=3, rest=4),
            code="AlexandreMotorImagery",
            interval=[0, 3],
            paradigm="imagery",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raw = Raw(self.data_path(subject), preload=True, verbose="ERROR")
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        url = "{:s}subject{:d}.raw.fif".format(ALEX_URL, subject)
        return dl.data_dl(url, "ALEXEEG", path, force_update, verbose)
