"""Typed sandbox exceptions."""


class SandboxUnavailable(RuntimeError):
    """Raised when required sandbox isolation is unavailable.

    Carries no extra fields beyond RuntimeError. Caller should propagate
    rather than catch - failing closed is the entire point.
    """
