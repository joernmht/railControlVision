"""Issue codes and validation report types shared by the validator, CLI and harness."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field


class IssueCode(StrEnum):
    """Closed set of validation issue codes (values equal the member names)."""

    SCHEMA_INVALID = "SCHEMA_INVALID"
    ID_DUPLICATE = "ID_DUPLICATE"
    TOPO_DANGLING_REF = "TOPO_DANGLING_REF"
    TOPO_PORT_UNKNOWN = "TOPO_PORT_UNKNOWN"
    TOPO_PORT_DUP = "TOPO_PORT_DUP"
    TOPO_DEGREE = "TOPO_DEGREE"
    TOPO_SELF_LOOP = "TOPO_SELF_LOOP"
    STATE_UNKNOWN_ELEMENT = "STATE_UNKNOWN_ELEMENT"
    STATE_WRONG_FIELD_FOR_KIND = "STATE_WRONG_FIELD_FOR_KIND"
    STATE_PATH_NOT_ALLOWED = "STATE_PATH_NOT_ALLOWED"
    STATE_MISSING = "STATE_MISSING"
    ROUTE_START_NOT_SIGNAL = "ROUTE_START_NOT_SIGNAL"
    ROUTE_DISCONTIGUOUS = "ROUTE_DISCONTIGUOUS"
    ROUTE_INVALID_TRAVERSAL = "ROUTE_INVALID_TRAVERSAL"
    ROUTE_SWITCH_MISMATCH = "ROUTE_SWITCH_MISMATCH"
    ROUTE_SWITCH_MISSING = "ROUTE_SWITCH_MISSING"
    GEOM_OUT_OF_BOUNDS = "GEOM_OUT_OF_BOUNDS"
    NODE_HALF_LABELS_UNEXPECTED = "NODE_HALF_LABELS_UNEXPECTED"


Severity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    """One finding of the validator."""

    model_config = ConfigDict(extra="forbid")

    code: IssueCode
    severity: Severity
    message: str
    element_id: str | None = None
    path: str | None = Field(
        default=None, description="Slash-joined location, e.g. 'topology/edges/2/a'."
    )


class ValidationReport(BaseModel):
    """Outcome of validating one document."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues with severity ``error``."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Issues with severity ``warning``."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    @classmethod
    def from_issues(cls, issues: Sequence[ValidationIssue]) -> ValidationReport:
        """Build a report; it is ok when no issue is an error.

        Args:
            issues: The issues found, in their final order.

        Returns:
            The report.
        """
        return cls(ok=all(issue.severity != "error" for issue in issues), issues=list(issues))


# Warnings in the default mode that strict mode reports as errors.
STRICT_PROMOTED: Final[frozenset[IssueCode]] = frozenset(
    {IssueCode.STATE_MISSING, IssueCode.GEOM_OUT_OF_BOUNDS, IssueCode.ROUTE_SWITCH_MISSING}
)
# Every code that is ever emitted as a warning; all other codes are always errors.
WARNING_CODES: Final[frozenset[IssueCode]] = STRICT_PROMOTED | {
    IssueCode.NODE_HALF_LABELS_UNEXPECTED
}
