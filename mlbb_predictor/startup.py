"""Small presentation helpers for first-open collection outcomes."""


def startup_notice(result: dict) -> tuple[str, str]:
    """Do not confuse a successful schedule check with fresh match downloads."""
    state = result.get("state")
    if result.get("profile_report", {}).get("state") in {"error", "partial"}:
        return "warning", "Some official team profiles could not refresh. Saved profiles remain available; see Data collection for source errors and retries."
    if state == "paused":
        return "info", "Automatic collection is paused. Loaded saved match data."
    if state == "timeout":
        return "info", "The data check is still running. Showing saved data for now; the predictor refreshes when new data is ready."
    if state in {"error", "busy", "stopped"}:
        return "warning", "The opening data check could not finish. Loaded saved match data instead. See Data collection for status and retry details."
    if state == "success":
        report = result.get("last_collection_report", {})
        # Older reports may be carried forward by a discovery-only check.
        if report and report.get("at") == result.get("last_checked_at"):
            if report.get("state") != "success":
                return "warning", "The schedule was checked, but some due match data could not be collected. Loaded the latest saved snapshot; see Data collection for details."
            if not result.get("cached") and report.get("new_games", 0):
                return "caption", f"Data ready. Collected {int(report['new_games'])} new games before loading the predictor."
        prefix = "Using a recent schedule check" if result.get("cached") else "Schedule checked"
        return "caption", f"{prefix}. Loaded saved ratings and hero pools. The 24-hour completion waiting rule still applies."
    return "warning", "Loaded saved match data; the opening check returned an unknown status."
