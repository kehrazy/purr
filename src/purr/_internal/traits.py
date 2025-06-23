"""Internal marker classes for operation traits."""


class Trait:
    """Base class for all operation traits."""

    pass


class NoSideEffect(Trait):
    """Indicates that an operation has no side effects."""

    pass
