"""Typed errors for the anecho SDK (mirrors a commercial SDK's error surface)."""
from __future__ import annotations


class AnechoError(Exception):
    """Base class for all anecho SDK errors."""


class ModelInvalidError(AnechoError):
    """The model file is malformed, truncated, or fails its integrity check."""


class ModelVersionUnsupportedError(AnechoError):
    """The container format version is not supported by this SDK build."""


class LicenseError(AnechoError):
    """Base class for license problems."""


class LicenseFormatInvalidError(LicenseError):
    """The license token is malformed or its signature does not verify."""


class LicenseExpiredError(LicenseError):
    """The license is valid but outside its validity window."""


class ProcessingNotAllowedError(LicenseError):
    """The license does not grant the requested feature / product."""


class AudioConfigUnsupportedError(AnechoError):
    """The requested sample rate or block size is not supported by the model."""
