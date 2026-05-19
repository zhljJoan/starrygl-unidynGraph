from __future__ import annotations


class ATCStarryError(Exception):
    """Base exception for ATC-StarryglLib."""


class ConfigError(ATCStarryError):
    """Raised when configuration is missing required fields or has invalid values."""


class RegistryError(ATCStarryError):
    """Raised when resolving or registering named components fails."""


class ArtifactError(ATCStarryError):
    """Raised when an expected runtime artifact is missing or invalid."""


class BackendError(ATCStarryError):
    """Raised when a backend cannot prepare, build, or iterate data."""
