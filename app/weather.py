"""
Weather API client for Open-Meteo.

All weather data fetching goes through this module. The graph nodes call
these functions — they never construct Open-Meteo URLs directly.

Design decisions:
- Uses requests (sync) for simplicity. Open-Meteo responds in <200ms typically.
- Geocoding and forecast are separate functions so they can fail independently
  (and the graph can route to different failure handlers).
- We always request a broad set of weather fields (current + daily + hourly)
  so that any SOP can be evaluated without needing to re-fetch. The cost is
  a slightly larger response payload (~5KB), which is negligible.
- Timeout is set to 10 seconds — generous enough for slow connections,
  short enough to not hang the user experience.
"""

import requests
from typing import Optional


# Open-Meteo endpoints (free, no API key required)
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Timeout for API calls (seconds)
API_TIMEOUT = 10

# Weather fields to request
CURRENT_FIELDS = ",".join([
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "precipitation_probability",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "uv_index",
])

DAILY_FIELDS = ",".join([
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "uv_index_max",
    "weather_code",
])

HOURLY_FIELDS = ",".join([
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "uv_index",
    "weather_code",
])


def geocode_city(city_name: str) -> dict:
    """Resolve a city name to latitude/longitude via Open-Meteo geocoding.

    Args:
        city_name: The city or place name to look up.

    Returns:
        Dict with keys: latitude, longitude, name, admin1, country, resolved_name.

    Raises:
        ValueError:  If no results are returned (city not found).
        RuntimeError: If the API request fails entirely.
    """
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city_name, "count": 5, "language": "en"},
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Geocoding API request failed: {e}")

    results = data.get("results")
    if not results or len(results) == 0:
        raise ValueError(
            f"Could not find a location matching '{city_name}'. "
            f"Please try a more specific name or check the spelling."
        )

    # Take the first result (most relevant match).
    # This is a reasonable default for common city names.
    top = results[0]
    admin1 = top.get("admin1", "")
    country = top.get("country", "")

    parts = [top["name"]]
    if admin1:
        parts.append(admin1)
    if country:
        parts.append(country)

    return {
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "name": top["name"],
        "admin1": admin1,
        "country": country,
        "resolved_name": ", ".join(parts),
    }


def fetch_weather(latitude: float, longitude: float) -> dict:
    """Fetch current, hourly, and daily weather data from Open-Meteo.

    Args:
        latitude:  Location latitude.
        longitude: Location longitude.

    Returns:
        Full API response dict with "current", "daily", and "hourly" sections.

    Raises:
        RuntimeError: If the API request fails or returns no usable data.
    """
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": CURRENT_FIELDS,
                "daily": DAILY_FIELDS,
                "hourly": HOURLY_FIELDS,
                "timezone": "auto",
                "forecast_days": 2,
            },
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Weather API request failed: {e}")

    # Validate that we actually got weather data, not just metadata
    if "current" not in data:
        raise RuntimeError(
            "Weather API returned metadata but no current weather data. "
            "This usually means the requested fields are invalid."
        )

    return data


def format_weather_summary(weather_data: dict) -> str:
    """Create a human-readable summary of current weather conditions.

    This is used in the bot's response so the user can see the actual numbers
    the advice is based on. Every value comes directly from the API response.
    """
    current = weather_data.get("current", {})
    units = weather_data.get("current_units", {})

    lines = []

    def add(label: str, key: str):
        val = current.get(key)
        if val is not None:
            unit = units.get(key, "")
            lines.append(f"  • {label}: {val} {unit}".strip())

    add("Temperature", "temperature_2m")
    add("Feels like", "apparent_temperature")
    add("Humidity", "relative_humidity_2m")
    add("Precipitation", "precipitation")
    add("Precipitation probability", "precipitation_probability")
    add("Wind speed", "wind_speed_10m")
    add("Wind gusts", "wind_gusts_10m")
    add("UV index", "uv_index")

    # Decode weather code to text
    wmo_code = current.get("weather_code")
    if wmo_code is not None:
        description = _weather_code_to_text(wmo_code)
        lines.append(f"  • Conditions: {description}")

    return "\n".join(lines) if lines else "  (No weather data available)"


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable description."""
    wmo_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return wmo_codes.get(code, f"Unknown (code {code})")
