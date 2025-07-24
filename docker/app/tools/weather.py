"""
Weather Tool - MVC Pattern Implementation

This tool provides weather information for a given location using the Open-Meteo API,
following the Model-View-Controller pattern.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import requests
from pydantic import BaseModel, Field
from tools.base import (
    BaseTool,
    BaseToolResponse,
    ExecutionMode,
    ToolController,
    ToolView,
)

logger = logging.getLogger(__name__)


class CurrentWeather(BaseModel):
    """Current weather conditions"""

    temperature: float = Field(description="Current temperature in Fahrenheit")
    relative_humidity: Optional[float] = Field(
        None, description="Current relative humidity in %"
    )
    wind_speed: float = Field(description="Current wind speed in km/h")
    weather_code: Optional[int] = Field(None, description="Weather condition code")
    is_day: Optional[bool] = Field(None, description="Whether it's day or night")
    precipitation_probability: Optional[float] = Field(
        None, description="Current precipitation probability in %"
    )


class HourlyWeather(BaseModel):
    """Hourly weather forecast"""

    time: List[str] = Field(default_factory=list, description="Hourly timestamps")
    temperature: List[float] = Field(
        default_factory=list, description="Hourly temperatures in Fahrenheit"
    )
    relative_humidity: List[float] = Field(
        default_factory=list, description="Hourly relative humidity in %"
    )
    wind_speed: List[float] = Field(
        default_factory=list, description="Hourly wind speed in km/h"
    )
    weather_code: List[int] = Field(
        default_factory=list, description="Hourly weather condition code"
    )
    is_day: List[bool] = Field(
        default_factory=list, description="Hourly whether it's day or night"
    )
    precipitation_probability: List[float] = Field(
        default_factory=list, description="Hourly precipitation probability in %"
    )


class WeatherResponse(BaseToolResponse):
    """Complete weather response from Open-Meteo API"""

    location: str = Field(description="Location name")
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    timezone: str = Field(description="Timezone")
    current: CurrentWeather = Field(description="Current weather conditions")
    hourly: Optional[HourlyWeather] = Field(None, description="Hourly weather forecast")
    source: str = Field(description="Source of the weather data")


class LocationResult(BaseModel):
    """Geocoding result for location lookup"""

    name: str
    latitude: float
    longitude: float
    country: str
    admin1: Optional[str] = None  # State/region
    admin2: Optional[str] = None  # County
    admin3: Optional[str] = None  # City district
    admin4: Optional[str] = None  # Suburb
    population: Optional[int] = None
    elevation: Optional[float] = None


class WeatherAPIClient:
    """Repository for weather API interactions"""

    def __init__(self):
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"

    def geocode_location(self, location: str) -> LocationResult:
        """Geocode a location string to coordinates"""
        params = {
            "name": location,
            "count": 10,
            "language": "en",
            "format": "json",
        }

        try:
            response = requests.get(self.geocoding_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                return LocationResult(
                    name=result.get("name", location),
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                    country=result.get("country", ""),
                    admin1=result.get("admin1"),
                    admin2=result.get("admin2"),
                    admin3=result.get("admin3"),
                    admin4=result.get("admin4"),
                    population=result.get("population"),
                    elevation=result.get("elevation"),
                )

            # If full location didn't work, try just the city
            city_only = self._extract_city_only(location)
            if city_only != location:
                params["name"] = city_only
                response = requests.get(self.geocoding_url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()

                if "results" in data and len(data["results"]) > 0:
                    result = data["results"][0]
                    return LocationResult(
                        name=result.get("name", city_only),
                        latitude=result["latitude"],
                        longitude=result["longitude"],
                        country=result.get("country", ""),
                        admin1=result.get("admin1"),
                        admin2=result.get("admin2"),
                        admin3=result.get("admin3"),
                        admin4=result.get("admin4"),
                        population=result.get("population"),
                        elevation=result.get("elevation"),
                    )

            raise ValueError(f"Location '{location}' not found")

        except requests.RequestException as e:
            logger.error(f"Geocoding request failed: {e}")
            raise ConnectionError(f"Failed to geocode location: {str(e)}")

    def fetch_weather(
        self, latitude: float, longitude: float, include_hourly: bool = True
    ) -> Dict[str, Any]:
        """Fetch weather data from Open-Meteo API"""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "is_day",
                "weather_code",
                "wind_speed_10m",
            ],
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "inch",
        }

        if include_hourly:
            params["hourly"] = [
                "temperature_2m",
                "relative_humidity_2m",
                "weather_code",
                "wind_speed_10m",
                "is_day",
                "precipitation_probability",
            ]
            params["forecast_days"] = 2

        try:
            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Weather request failed: {e}")
            raise ConnectionError(f"Failed to fetch weather data: {str(e)}")

    def _extract_city_only(self, location: str) -> str:
        """Extract just the city name from a location string"""
        parts = location.split(",")
        city = parts[0].strip()

        words = city.split()
        if len(words) > 1 and len(words[-1]) <= 3 and words[-1].isupper():
            city = " ".join(words[:-1])

        return city


class WeatherController(ToolController):
    """Controller handling weather business logic"""

    def __init__(self):
        self.api_client = WeatherAPIClient()

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the weather request"""
        location = params['location']
        include_hourly = params.get('include_hourly', True)

        # Geocode location
        location_result = self.api_client.geocode_location(location)

        # Get location display name
        display_name = self._get_display_name(location_result)

        # Fetch weather data
        weather_data = self.api_client.fetch_weather(
            location_result.latitude, location_result.longitude, include_hourly
        )

        # Parse current weather
        current_weather = self._parse_current_weather(weather_data)

        # Parse hourly weather if included
        hourly_weather = None
        if include_hourly and "hourly" in weather_data:
            hourly_weather = self._parse_hourly_weather(weather_data)

        return {
            "location": display_name,
            "latitude": location_result.latitude,
            "longitude": location_result.longitude,
            "timezone": weather_data.get("timezone", "UTC"),
            "current": current_weather,
            "hourly": hourly_weather,
            "source": "Open-Meteo API",
        }

    def _get_display_name(self, location: LocationResult) -> str:
        """Format location for display"""
        parts = [location.name]

        if location.admin1 and location.admin1 != location.name:
            parts.append(location.admin1)

        if location.country:
            parts.append(location.country)

        return ", ".join(parts)

    def _parse_current_weather(self, data: Dict[str, Any]) -> CurrentWeather:
        """Parse current weather from API response"""
        current = data.get("current", {})

        return CurrentWeather(
            temperature=current.get("temperature_2m", 0),
            relative_humidity=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m", 0),
            weather_code=current.get("weather_code"),
            is_day=current.get("is_day", 1) == 1,
            precipitation_probability=None,  # Not available in current weather
        )

    def _parse_hourly_weather(self, data: Dict[str, Any]) -> HourlyWeather:
        """Parse hourly weather forecast from API response"""
        hourly = data.get("hourly", {})

        # Convert is_day from 0/1 to boolean
        is_day_values = hourly.get("is_day", [])
        is_day_bools = [val == 1 for val in is_day_values]

        return HourlyWeather(
            time=hourly.get("time", []),
            temperature=hourly.get("temperature_2m", []),
            relative_humidity=hourly.get("relative_humidity_2m", []),
            wind_speed=hourly.get("wind_speed_10m", []),
            weather_code=hourly.get("weather_code", []),
            is_day=is_day_bools,
            precipitation_probability=hourly.get("precipitation_probability", []),
        )


class WeatherView(ToolView):
    """View for formatting weather responses"""

    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format raw weather data into WeatherResponse"""
        try:
            return WeatherResponse(**data)
        except Exception as e:
            logger.error(f"Error formatting weather response: {e}")
            return WeatherResponse(
                location="Unknown",
                latitude=0,
                longitude=0,
                timezone="UTC",
                current=CurrentWeather(temperature=0, wind_speed=0),
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
            )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into WeatherResponse"""
        error_code = "UNKNOWN_ERROR"
        error_message = str(error)

        if isinstance(error, ValueError):
            error_code = "LOCATION_NOT_FOUND"
        elif isinstance(error, ConnectionError):
            error_code = "API_ERROR"
        elif isinstance(error, TimeoutError):
            error_code = "TIMEOUT_ERROR"

        return WeatherResponse(
            location="Unknown",
            latitude=0,
            longitude=0,
            timezone="UTC",
            current=CurrentWeather(temperature=0, wind_speed=0),
            success=False,
            error_message=error_message,
            error_code=error_code,
        )


class WeatherTool(BaseTool):
    """Tool for getting weather information following MVC pattern"""

    def __init__(self):
        super().__init__()
        self.name = "get_weather"
        self.description = "Get current weather for a specific location. Use when user asks for weather, temperature, or forecast AND provides a city/location."
        self.execution_mode = ExecutionMode.SYNC
        self.timeout = 10.0

    def _initialize_mvc(self):
        """Initialize MVC components"""
        self._controller = WeatherController()
        self._view = WeatherView()

    def get_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country to get weather for (e.g., 'New York, NY' or 'London, UK')",
                        },
                        "but_why": {
                            "type": "integer",
                            "description": "An integer from 1-5 where a larger number indicates confidence this is the right tool to help the user.",
                        },
                    },
                    "required": ["location", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get the response type for this tool"""
        return WeatherResponse


# Helper functions for backward compatibility
def get_weather_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for weather lookup

    Returns:
        Dict containing the OpenAI tool definition
    """
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("get_weather", WeatherTool)

    # Get the tool instance and return its definition
    tool = get_tool("get_weather")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get weather tool definition")
