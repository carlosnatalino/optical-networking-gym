from __future__ import annotations

from enum import StrEnum


class TrafficMode(StrEnum):
    DYNAMIC = "dynamic"
    STATIC = "static"


class MaskMode(StrEnum):
    RESOURCE_ONLY = "resource_only"
    RESOURCE_AND_QOT = "resource_and_qot"


class RewardProfile(StrEnum):
    BALANCED = "balanced"
    LEGACY = "legacy"


class Status(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_BY_AGENT = "rejected_by_agent"
    BLOCKED_RESOURCES = "blocked_resources"
    BLOCKED_QOT = "blocked_qot"
