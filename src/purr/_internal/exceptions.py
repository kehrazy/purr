"""Internal custom exceptions for the Purr DSL."""


class PurrError(Exception):
    """Base exception for all errors raised by the Purr framework."""

    pass


class DefinitionError(PurrError):
    """
    Raised when an IR component definition is invalid.

    This error indicates a problem with how an operation or dialect was
    defined by the user, such as a missing field or a type mismatch.
    """

    pass
