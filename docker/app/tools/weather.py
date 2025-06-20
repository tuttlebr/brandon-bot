import logging
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class CurrentWeather(BaseModel):
    """Current weather conditions"""

    temperature: float = Field(description="Current temperature in Fahrenheit")
    relative_humidity: Optional[float] = Field(None, description="Current relative humidity in %")
    wind_speed: float = Field(description="Current wind speed in km/h")
    weather_code: Optional[int] = Field(None, description="Weather condition code")
    is_day: Optional[bool] = Field(None, description="Whether it's day or night")
    precipitation_probability: Optional[float] = Field(None, description="Precipitation probability in %")


class HourlyWeather(BaseModel):
    """Hourly weather forecast"""

    time: List[str] = Field(default_factory=list, description="Hourly timestamps")
    temperature: List[float] = Field(default_factory=list, description="Hourly temperatures in Fahrenheit")
    relative_humidity: List[float] = Field(default_factory=list, description="Hourly relative humidity in %")
    wind_speed: List[float] = Field(default_factory=list, description="Hourly wind speed in km/h")
    weather_code: List[int] = Field(default_factory=list, description="Hourly weather condition code")
    is_day: List[bool] = Field(default_factory=list, description="Hourly whether it's day or night")
    precipitation_probability: List[float] = Field(
        default_factory=list, description="Hourly precipitation probability in %"
    )


class WeatherResponse(BaseModel):
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


class WeatherTool:
    """Tool for getting weather information using Open-Meteo API"""

    def __init__(self):
        self.name = "get_weather"
        self.description = "Triggered when asks for weather or forecast related information. Data are provided by [Open-Meteo](https://open-meteo.com/). Input should be a location string of 'City' or 'ZIP code'. If input is in 'City, ST' format, only the city name will be used."
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        self.source = "[Open-Meteo](https://open-meteo.com/)"

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format

        Returns:
            Dict containing the OpenAI-compatible tool definition
        """
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
                            "description": "City name or ZIP code. If provided as 'City, ST', only the city name will be used for weather lookup.",
                        }
                    },
                    "required": ["location"],
                },
            },
        }

    def _celcius_to_fahrenheit(self, temperature: float) -> float:
        """
        Convert temperature from Celsius to Fahrenheit
        """
        return round((temperature * 9 / 5) + 32, 1)

    def _extract_city_only(self, location: str) -> str:
        """
        Extract just the city name from "City, ST" format input

        Args:
            location: Location string that may be in "City, ST" format

        Returns:
            str: Just the city name
        """
        # Remove periods first
        location = location.replace(".", "")

        # If there's a comma, take only the part before it (the city)
        if "," in location:
            city = location.split(",")[0].strip()
            logger.debug(f"Extracted city '{city}' from '{location}'")
            return city

        # If no comma, return the original location
        return location.strip()

    def _geocode_location(self, location: str) -> LocationResult:
        """
        Convert location string to coordinates using Open-Meteo Geocoding API

        Args:
            location: Location string like "City, Country"

        Returns:
            LocationResult: Geocoded location information

        Raises:
            ValueError: If location cannot be found
            requests.RequestException: If the API request fails
        """
        # Extract only the city name from "City, ST" format
        location = self._extract_city_only(location)
        logger.info(f"Geocoding location: '{location}'")

        try:
            params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
                "countryCode": "US",
            }

            response = requests.get(self.geocoding_url, params=params)
            response.raise_for_status()

            data = response.json()

            if not data.get("results"):
                logger.error(f"No results found for location: '{location}'")
                raise ValueError(f"Location '{location}' not found")

            result = data["results"][0]
            location_result = LocationResult(
                name=result["name"],
                latitude=result["latitude"],
                longitude=result["longitude"],
                country=result["country"],
                admin1=result.get("admin1"),
            )

            logger.info(
                f"Successfully geocoded '{location}' to {location_result.latitude}, {location_result.longitude}"
            )
            return location_result

        except requests.exceptions.RequestException as e:
            logger.error(f"Geocoding API request failed: {e}")
            raise requests.RequestException(f"Failed to geocode location: {str(e)}")

    def get_weather(self, location: str, include_hourly: bool = True) -> WeatherResponse:
        """
        Get weather information for a given location

        Args:
            location: Location string like "City, Country"
            include_hourly: Whether to include hourly forecast data

        Returns:
            WeatherResponse: The weather information in a validated Pydantic model

        Raises:
            ValueError: If location cannot be found
            requests.RequestException: If the API request fails
        """
        logger.info(f"Getting weather for location: '{location}'")

        # First, geocode the location
        try:
            location_info = self._geocode_location(location)
        except ValueError as e:
            logger.error(f"Error geocoding location: {e}")
            location_info = self._geocode_location("48176")

        # Prepare weather API parameters
        params = {
            "latitude": location_info.latitude,
            "longitude": location_info.longitude,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code,precipitation_probability,is_day",
            "timezone": "auto",
        }

        if include_hourly:
            params[
                "hourly"
            ] = "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code,precipitation_probability,is_day"
            params["forecast_days"] = 14  # Only include today's hourly forecast

        try:
            logger.debug(
                f"Making weather API request for coordinates: {location_info.latitude}, {location_info.longitude}"
            )

            response = requests.get(self.weather_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Parse current weather
            current_data = data["current"]
            current_weather = CurrentWeather(
                temperature=self._celcius_to_fahrenheit(current_data["temperature_2m"]),
                wind_speed=current_data["wind_speed_10m"],
                relative_humidity=current_data.get("relative_humidity_2m"),
                weather_code=current_data.get("weather_code"),
                precipitation_probability=current_data.get("precipitation_probability"),
                is_day=(current_data.get("is_day") == 1 if current_data.get("is_day") is not None else None),
            )

            # Parse hourly weather if requested
            hourly_weather = None
            if include_hourly and "hourly" in data:
                hourly_data = data["hourly"]
                hourly_weather = HourlyWeather(
                    time=hourly_data.get("time", []),
                    temperature=[self._celcius_to_fahrenheit(temp) for temp in hourly_data.get("temperature_2m", [])],
                    relative_humidity=hourly_data.get("relative_humidity_2m", []),
                    wind_speed=hourly_data.get("wind_speed_10m", []),
                    precipitation_probability=hourly_data.get("precipitation_probability", []),
                    weather_code=hourly_data.get("weather_code", []),
                    is_day=hourly_data.get("is_day", []),
                )

            # Create the response
            weather_response = WeatherResponse(
                location=f"{location_info.name}, {location_info.country}",
                latitude=location_info.latitude,
                longitude=location_info.longitude,
                timezone=data.get("timezone", "UTC"),
                current=current_weather,
                hourly=hourly_weather,
                source=self.source,
            )

            logger.info(
                f"Weather data retrieved successfully for '{location}'. Current temperature: {current_weather.temperature}°F"
            )
            return weather_response

        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
            raise requests.RequestException(f"Failed to get weather data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during weather lookup: {e}")
            raise

    def _run(self, location: str = None, **kwargs) -> WeatherResponse:
        """
        Execute a weather lookup with the given location.

        Args:
            location: The location string (for backward compatibility)
            **kwargs: Can accept a dictionary with 'location' key

        Returns:
            WeatherResponse: The weather information in a validated Pydantic model
        """
        # Support both direct parameter and dictionary input
        if location is None and "location" in kwargs:
            location = kwargs["location"]
        elif location is None:
            raise ValueError("Location parameter is required")

        logger.debug(f"_run method called with location: '{location}'")
        return self.get_weather(location)

    def run_with_dict(self, params: Dict[str, Any]) -> WeatherResponse:
        """
        Execute a weather lookup with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'location'

        Returns:
            WeatherResponse: The weather information in a validated Pydantic model
        """
        if "location" not in params:
            raise ValueError("'location' key is required in parameters dictionary")

        location = params["location"]
        logger.debug(f"run_with_dict method called with location: '{location}'")
        return self.get_weather(location)


# Create a global instance and helper functions for easy access
weather_tool = WeatherTool()


def get_weather_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for weather lookup

    Returns:
        Dict containing the OpenAI tool definition
    """
    return weather_tool.to_openai_format()


def execute_weather_search(location: str, include_hourly: bool = False) -> WeatherResponse:
    """
    Execute a weather lookup with the given location

    Args:
        location: The location string
        include_hourly: Whether to include hourly forecast data

    Returns:
        WeatherResponse: The weather information
    """
    return weather_tool.get_weather(location, include_hourly)


def execute_weather_with_dict(params: Dict[str, Any]) -> WeatherResponse:
    """
    Execute a weather lookup with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'location'

    Returns:
        WeatherResponse: The weather information
    """
    return weather_tool.run_with_dict(params)
