from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class Violation:
    scanner: str
    details: Optional[str] = None
    action: Optional[str] = None


@dataclass
class GuardrailResult:
    result: Literal["PASS", "BLOCKED", "SANITIZED"]
    risk_score: float = 0.0
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"
    sanitized_text: Optional[str] = None
    violations: list[Violation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardrailResult:
        violations = [
            Violation(
                scanner=v.get("scanner", ""),
                details=v.get("details"),
                action=v.get("action"),
            )
            for v in data.get("violations") or []
        ]
        return cls(
            result=data.get("result", "PASS"),
            risk_score=data.get("riskScore", 0.0),
            risk_level=data.get("riskLevel", "LOW"),
            sanitized_text=data.get("sanitizedText"),
            violations=violations,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "riskScore": self.risk_score,
            "riskLevel": self.risk_level,
        }
