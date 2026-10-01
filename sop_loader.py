"""
SOP Loader and Evaluator.

This module handles:
1. Loading SOPs from the YAML file (done once at import, cached).
2. Evaluating weather conditions against SOP thresholds (pure deterministic logic).

Design decisions:
- SOPs are loaded from a YAML file on disk, not hardcoded. The path is relative
  to the project root so the policy team can edit sops.yaml without touching code.
- Condition evaluation is 100% deterministic Python — no LLM involvement.
  The LLM never decides whether a threshold is met; code does that. This ensures
  the bot cannot misreport whether conditions are dangerous.
- The "between" operator is inclusive on both ends: value[0] <= x <= value[1].
- Daily parameters are supported by prefixing with "daily." (e.g., "daily.precipitation_sum").
  The evaluator knows to look in the daily data section of the weather response.
"""

import os
import yaml
from typing import Any


# ---------------------------------------------------------------------------
# SOP Loading
# ---------------------------------------------------------------------------

_sops_cache: list[dict] | None = None
_sops_file_path: str | None = None


def _default_sops_path() -> str:
    """Return the default path to sops.yaml (project root)."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "sops.yaml")


def load_sops(filepath: str | None = None, force_reload: bool = False) -> list[dict]:
    """Load SOPs from YAML file. Results are cached after first load.

    Args:
        filepath:      Path to the YAML file. Defaults to project-root/sops.yaml.
        force_reload:  If True, bypass cache and re-read from disk.

    Returns:
        List of SOP dictionaries.

    Raises:
        FileNotFoundError: If the SOP file doesn't exist.
        yaml.YAMLError:    If the file has invalid YAML.
    """
    global _sops_cache, _sops_file_path

    if filepath is None:
        filepath = _default_sops_path()

    if _sops_cache is not None and _sops_file_path == filepath and not force_reload:
        return _sops_cache

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _sops_cache = data.get("sops", [])
    _sops_file_path = filepath
    return _sops_cache


def get_sop_summary() -> str:
    """Return a compact text summary of all loaded SOPs for LLM context.

    This is injected into the response-composition prompt so the LLM knows
    what SOPs exist without needing to see the full YAML.
    """
    sops = load_sops()
    lines = []
    for sop in sops:
        lines.append(
            f"- {sop['id']} ({sop['name']}): "
            f"category={sop['category']}, severity={sop['severity']}, "
            f"activities={sop['activities']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Condition Evaluation (Deterministic — No LLM)
# ---------------------------------------------------------------------------

def _get_weather_value(weather_data: dict, parameter: str) -> float | None:
    """Extract a weather parameter value from the API response.

    Supports dotted paths for daily data:
      - "temperature_2m"          → weather_data["current"]["temperature_2m"]
      - "daily.precipitation_sum" → weather_data["daily"]["precipitation_sum"][0]
        (index 0 = today)

    Returns None if the parameter is missing (which causes the condition to
    evaluate as False, not as an error — a missing parameter simply means
    we can't confirm the condition is met).
    """
    if parameter.startswith("daily."):
        daily_param = parameter[len("daily."):]
        daily = weather_data.get("daily", {})
        values = daily.get(daily_param)
        if values and len(values) > 0:
            return values[0]  # Today's value
        return None
    else:
        current = weather_data.get("current", {})
        return current.get(parameter)


def _evaluate_single_check(weather_data: dict, check: dict) -> bool:
    """Evaluate a single condition check against weather data.

    Args:
        weather_data: Full weather API response with "current" and "daily" keys.
        check:        A dict with keys: parameter, operator, value.

    Returns:
        True if the condition is met, False otherwise.
        Returns False if the weather parameter is missing (conservative: we don't
        trigger warnings based on data we don't have).
    """
    parameter = check["parameter"]
    op = check["operator"]
    threshold = check["value"]

    actual = _get_weather_value(weather_data, parameter)
    if actual is None:
        return False

    if op == ">":
        return actual > threshold
    elif op == ">=":
        return actual >= threshold
    elif op == "<":
        return actual < threshold
    elif op == "<=":
        return actual <= threshold
    elif op == "==":
        return actual == threshold
    elif op == "between":
        if isinstance(threshold, list) and len(threshold) == 2:
            return threshold[0] <= actual <= threshold[1]
        return False
    else:
        # Unknown operator — don't trigger the SOP
        return False


def evaluate_sop_conditions(sop: dict, weather_data: dict) -> bool:
    """Evaluate whether a SOP's conditions are met by the given weather data.

    Uses the SOP's `logic` field to determine AND ("all") vs OR ("any") combination.

    Args:
        sop:          A single SOP dictionary from the loaded YAML.
        weather_data: Full weather API response.

    Returns:
        True if the SOP's conditions are satisfied.
    """
    conditions = sop.get("conditions", {})
    checks = conditions.get("checks", [])
    logic = conditions.get("logic", "all")

    if not checks:
        return False

    results = [_evaluate_single_check(weather_data, c) for c in checks]

    if logic == "all":
        return all(results)
    elif logic == "any":
        return any(results)
    else:
        return all(results)  # Default to AND


def get_matched_condition_details(sop: dict, weather_data: dict) -> list[dict]:
    """For a matched SOP, return details about which conditions triggered.

    This is used for traceability — the bot can tell the user exactly which
    weather values triggered which thresholds.

    Returns:
        List of dicts with: parameter, operator, threshold, actual_value, triggered.
    """
    conditions = sop.get("conditions", {})
    checks = conditions.get("checks", [])
    details = []

    for check in checks:
        param = check["parameter"]
        actual = _get_weather_value(weather_data, param)
        triggered = _evaluate_single_check(weather_data, check)
        details.append({
            "parameter": param,
            "operator": check["operator"],
            "threshold": check["value"],
            "actual_value": actual,
            "triggered": triggered,
        })

    return details


def match_sops_against_weather(
    weather_data: dict,
    user_activities: list[str],
) -> list[dict]:
    """Match all SOPs against current weather data and user activities.

    Process:
    1. Load all SOPs from YAML.
    2. For each SOP, check if the user's activity matches the SOP's activity list.
    3. If activity matches, evaluate the SOP's weather conditions.
    4. Collect all matching SOPs with their trigger details.
    5. Sort by severity (critical first, info last).

    Args:
        weather_data:    Full weather API response.
        user_activities: Standardized activity tags from intent parsing.

    Returns:
        List of matched SOP dicts, each augmented with "matched_conditions"
        showing which thresholds triggered. Sorted by severity.
    """
    sops = load_sops()
    severity_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "info": 4}
    matched = []

    for sop in sops:
        # Check activity match
        sop_activities = sop.get("activities", [])
        if "*" not in sop_activities:
            # Check if any user activity matches any SOP activity
            if not any(act in sop_activities for act in user_activities):
                continue

        # Check weather conditions
        if evaluate_sop_conditions(sop, weather_data):
            condition_details = get_matched_condition_details(sop, weather_data)
            matched.append({
                "id": sop["id"],
                "name": sop["name"],
                "category": sop["category"],
                "severity": sop["severity"],
                "guidance": sop["guidance"].strip(),
                "matched_conditions": condition_details,
            })

    # Sort by severity: critical first
    matched.sort(key=lambda x: severity_order.get(x["severity"], 5))

    return matched
