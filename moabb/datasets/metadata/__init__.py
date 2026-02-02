"""Metadata schema module for MOABB datasets.

This module provides standardized dataclasses for documenting dataset metadata.
Metadata is distributed across individual dataset files as class attributes.

Core Classes
------------
AcquisitionMetadata
    Technical acquisition parameters (sampling rate, channels, hardware, etc.)
DocumentationMetadata
    Publication and dataset provenance information (DOI, authors, repository)
ParticipantMetadata
    Subject demographics (sample size, age, gender, health status)
ExperimentMetadata
    Paradigm and task details (events, trial structure, task type)
DatasetMetadata
    Top-level container combining all metadata sections

Additional Classes
------------------
Demographics
    Extended subject demographics (subjects_count, ages, age_min, age_max)
ExternalLinks
    URLs and data source links
Timestamps
    Dataset creation and modification dates
Tags
    Classification tags
ChannelCount
    Channel count distribution entry
SamplingRateCount
    Sampling rate distribution entry

New Classes (from RALPH extraction)
-----------------------------------
AuxiliaryChannelsMetadata
    EOG, EMG, and other physiological channel information
FilterDetails
    Filter configuration details (highpass, lowpass, notch, etc.)
PreprocessingMetadata
    Preprocessing and artifact handling details
FrequencyBands
    Frequency band definitions for analysis
SignalProcessingMetadata
    Feature extraction and classification methods
CrossValidationMetadata
    Cross-validation methodology details
PerformanceMetadata
    Reported performance metrics
BCIApplicationMetadata
    BCI application context and environment
ParadigmSpecificMetadata
    Paradigm-specific parameters (SSVEP frequencies, c-VEP codes, etc.)
DataStructureMetadata
    Data organization and trial structure

Functions
---------
validate_country_code
    Validate ISO 3166-1 alpha-2 country codes
validate_metadata_against_dataset
    Validate metadata matches actual dataset structure
get_dataset_description
    Extract description from dataset class docstring

Example
-------
>>> from moabb.datasets.metadata import (
...     DatasetMetadata, AcquisitionMetadata,
...     ParticipantMetadata, ExperimentMetadata
... )
>>> metadata = DatasetMetadata(
...     acquisition=AcquisitionMetadata(
...         sampling_rate=512.0,
...         n_channels=64,
...         channel_types={"eeg": 60, "eog": 4},
...     ),
...     participants=ParticipantMetadata(n_subjects=20),
...     experiment=ExperimentMetadata(paradigm="imagery"),
... )

>>> # Access metadata from a dataset class
>>> from moabb.datasets import BNCI2014_001
>>> print(BNCI2014_001.METADATA.participants.n_subjects)
9
"""

from .schema import (  # Core MOABB classes; Additional classes; New classes from RALPH extraction; Validation functions
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    ChannelCount,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    Demographics,
    DocumentationMetadata,
    ExperimentMetadata,
    ExternalLinks,
    FilterDetails,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
    PreprocessingMetadata,
    SamplingRateCount,
    SignalProcessingMetadata,
    Tags,
    Timestamps,
    get_dataset_description,
    validate_country_code,
    validate_metadata_against_dataset,
)


__all__ = [
    # Core MOABB classes
    "AcquisitionMetadata",
    "DocumentationMetadata",
    "ParticipantMetadata",
    "ExperimentMetadata",
    "DatasetMetadata",
    # Additional classes
    "Demographics",
    "ExternalLinks",
    "Timestamps",
    "Tags",
    "ChannelCount",
    "SamplingRateCount",
    # New classes from RALPH extraction
    "AuxiliaryChannelsMetadata",
    "FilterDetails",
    "PreprocessingMetadata",
    "FrequencyBands",
    "SignalProcessingMetadata",
    "CrossValidationMetadata",
    "PerformanceMetadata",
    "BCIApplicationMetadata",
    "ParadigmSpecificMetadata",
    "DataStructureMetadata",
    # Validation functions
    "validate_country_code",
    "validate_metadata_against_dataset",
    "get_dataset_description",
]
