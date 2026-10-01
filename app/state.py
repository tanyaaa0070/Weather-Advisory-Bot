"""
Graph state definition for the Weather Advisory Bot.

Design decision: We use a TypedDict (not a Pydantic model) because LangGraph's
StateGraph expects TypedDict for state schema. The state is the single source of
truth flowing through all graph nodes — each node reads what it needs and writes
its outputs back.

The `messages` field uses the `add` reducer so nodes append to the conversation
history rather than overwriting it. All other fields use default overwrite semantics.
"""

from typing import TypedDict, Optional, Annotated
import operator


class AdvisoryState(TypedDict):
    """State that flows through the LangGraph advisory pipeline.

    Attributes:
        messages:           Full conversation history (user + assistant turns).
                            Uses append semantics via operator.add reducer.
        current_query:      The user's current raw message text.

        # --- Intent parsing outputs ---
        location_name:      Extracted city/location name from user query.
        activities:         Standardized activity tags extracted from the query.
        time_context:       When the user is asking about (e.g., "today", "evening").
        is_weather_query:   Whether the query is actually about weather/outdoor safety.

        # --- Geocoding outputs ---
        latitude:           Resolved latitude from geocoding API.
        longitude:          Resolved longitude from geocoding API.
        resolved_location:  Full resolved place name (city, region, country).

        # --- Weather data ---
        weather_data:       Raw weather API response (current + daily + hourly).

        # --- SOP matching ---
        matched_sops:       List of SOPs whose conditions matched, sorted by severity.

        # --- Final output ---
        response:           The composed response text to return to the user.
        error:              Error message if any step failed.
    """
    messages: Annotated[list, operator.add]
    current_query: str

    # Intent
    location_name: Optional[str]
    activities: list
    time_context: Optional[str]
    is_weather_query: bool

    # Geocoding
    latitude: Optional[float]
    longitude: Optional[float]
    resolved_location: Optional[str]

    # Weather
    weather_data: Optional[dict]

    # SOP matching
    matched_sops: list

    # Output
    response: str
    error: Optional[str]
