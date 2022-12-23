
def get_milliseconds(duration: float, unit: str) -> float:
    if unit == "msecond":
        return duration
    if unit == "usecond":
        return duration / 1000
    raise ValueError(f"bad unit: {unit}")
