"""
Weather API client for Open-Meteo with automatic fallback.

All weather data fetching goes through this module. The graph nodes call
these functions — they never construct Open-Meteo URLs directly.

Resilience:
- Sends custom User-Agent identifying the application to avoid generic cloud blocks.
- If Open-Meteo returns HTTP 429 (rate limit on shared cloud IPs like Render)
  or fails, automatically falls back to secondary live weather endpoints.
"""

import requests
from typing import Optional


# Open-Meteo endpoints
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Headers identifying the application
DEFAULT_HEADERS = {
    "User-Agent": "WeatherAdvisoryBot/1.0 (https://github.com/tanyaaa0070/Weather-Advisory-Bot; contact: tanyaaa0070@gmail.com)",
    "Accept": "application/json",
}

# Timeout for API calls (seconds)
API_TIMEOUT = 8

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


def _geocode_fallback(city_name: str) -> dict:
    """Fallback geocoding using wttr.in when Open-Meteo rate-limits cloud IPs."""
    url = f"https://wttr.in/{city_name}?format=j1"
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    areas = data.get("nearest_area", [])
    if not areas:
        raise ValueError(f"Could not find a location matching '{city_name}'.")

    area = areas[0]
    lat = float(area.get("latitude", 0))
    lon = float(area.get("longitude", 0))
    area_name = area.get("areaName", [{}])[0].get("value", city_name)
    country = area.get("country", [{}])[0].get("value", "")
    region = area.get("region", [{}])[0].get("value", "")

    parts = [area_name]
    if region and region.lower() != area_name.lower():
        parts.append(region)
    if country:
        parts.append(country)

    return {
        "latitude": lat,
        "longitude": lon,
        "name": area_name,
        "admin1": region,
        "country": country,
        "resolved_name": ", ".join(parts),
    }


def geocode_city(city_name: str) -> dict:
    """Resolve a city name to latitude/longitude via Open-Meteo geocoding."""
    city_name = city_name.strip()
    if not city_name:
        raise ValueError("City name cannot be empty.")

    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city_name, "count": 5, "language": "en"},
            headers=DEFAULT_HEADERS,
            timeout=API_TIMEOUT,
        )
        if resp.status_code == 429:
            return _geocode_fallback(city_name)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if not results or len(results) == 0:
            return _geocode_fallback(city_name)

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
    except Exception as e:
        try:
            return _geocode_fallback(city_name)
        except Exception:
            raise RuntimeError(f"Geocoding request failed: {e}")


def _fetch_weather_fallback(latitude: float, longitude: float) -> dict:
    """Fallback weather fetch using wttr.in when Open-Meteo is rate-limited on shared IPs."""
    url = f"https://wttr.in/{latitude},{longitude}?format=j1"
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    curr = data.get("current_condition", [{}])[0]
    weather_days = data.get("weather", [{}])
    weather_day = weather_days[0] if weather_days else {}

    temp = float(curr.get("temp_C", 25.0))
    feels_like = float(curr.get("FeelsLikeC", temp))
    humidity = float(curr.get("humidity", 50.0))
    precip = float(curr.get("precipMM", 0.0))
    wind_kmph = float(curr.get("windspeedKmph", 10.0))
    uv = float(curr.get("uvIndex", 1.0))

    weather_desc = curr.get("weatherDesc", [{}])[0].get("value", "").lower()
    weather_code = 0
    if "rain" in weather_desc or "drizzle" in weather_desc:
        weather_code = 61
    elif "cloud" in weather_desc or "overcast" in weather_desc:
        weather_code = 3

    max_temp = float(weather_day.get("maxtempC", temp))
    min_temp = float(weather_day.get("mintempC", temp))
    hourly_items = weather_day.get("hourly", [])
    rain_chance = float(hourly_items[0].get("chanceofrain", 0.0)) if hourly_items else 0.0

    return {
        "current": {
            "temperature_2m": temp,
            "apparent_temperature": feels_like,
            "relative_humidity_2m": humidity,
            "precipitation": precip,
            "precipitation_probability": rain_chance,
            "rain": precip,
            "wind_speed_10m": wind_kmph,
            "wind_gusts_10m": round(wind_kmph * 1.25, 1),
            "uv_index": uv,
            "weather_code": weather_code,
        },
        "current_units": {
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "precipitation_probability": "%",
            "wind_speed_10m": "km/h",
            "wind_gusts_10m": "km/h",
            "uv_index": "",
            "weather_code": "",
        },
        "daily": {
            "temperature_2m_max": [max_temp],
            "temperature_2m_min": [min_temp],
            "precipitation_sum": [precip],
            "precipitation_probability_max": [rain_chance],
            "wind_speed_10m_max": [wind_kmph],
            "wind_gusts_10m_max": [round(wind_kmph * 1.25, 1)],
            "uv_index_max": [uv],
            "weather_code": [weather_code],
        },
        "hourly": {
            "temperature_2m": [temp],
            "precipitation_probability": [rain_chance],
            "precipitation": [precip],
            "wind_speed_10m": [wind_kmph],
            "wind_gusts_10m": [round(wind_kmph * 1.25, 1)],
            "uv_index": [uv],
            "weather_code": [weather_code],
        },
    }


def fetch_weather(latitude: float, longitude: float) -> dict:
    """Fetch current, hourly, and daily weather data with automatic failover."""
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
            headers=DEFAULT_HEADERS,
            timeout=API_TIMEOUT,
        )
        if resp.status_code == 429:
            return _fetch_weather_fallback(latitude, longitude)
        resp.raise_for_status()
        data = resp.json()
        if "current" not in data:
            return _fetch_weather_fallback(latitude, longitude)
        return data
    except Exception as e:
        try:
            return _fetch_weather_fallback(latitude, longitude)
        except Exception:
            raise RuntimeError(f"Weather API request failed: {e}")


def format_weather_summary(weather_data: dict) -> str:
    """Create a human-readable summary of current weather conditions."""
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
