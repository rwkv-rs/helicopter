class ContractError(ValueError):
    """The source, target, or run violates the frozen conversion contract."""


class CoverageError(ContractError):
    """A source or target tensor lacks an explicit mapping disposition."""


class CompatibilityError(ContractError):
    """A pinned loader or checkpoint layout is incompatible with the run."""

