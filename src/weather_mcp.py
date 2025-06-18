# Import required libraries
from typing import Any  # For type hints
import httpx  # For making async HTTP requests
from mcp.server.fastmcp import FastMCP  # MCP server implementation

# Initialize the MCP server with a name "weather"
# This name will be used to identify the server in the MCP ecosystem
mcp = FastMCP("weather")

# Constants for the National Weather Service API
NWS_API_BASE = "https://api.weather.gov"  # Base URL for the weather API
USER_AGENT = "weather-app/1.0"  # User agent for API requests

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling.
    
    This helper function handles all HTTP requests to the National Weather Service API.
    It includes proper headers and error handling to ensure robust API communication.
    
    Args:
        url: The complete URL to make the request to
        
    Returns:
        dict: JSON response from the API if successful
        None: If the request fails for any reason
    """
    headers = {
        "User-Agent": USER_AGENT,  # Required by NWS API
        "Accept": "application/geo+json"  # Request GeoJSON format
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string.
    
    Takes a weather alert feature from the NWS API and formats it into
    a human-readable string with all relevant information.
    
    Args:
        feature: A weather alert feature from the NWS API
        
    Returns:
        str: Formatted string containing alert details
    """
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@mcp.tool()  # Decorator to register this function as an MCP tool
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.
    
    This tool fetches active weather alerts for a specified US state
    using the National Weather Service API.
    
    Args:
        state: Two-letter US state code (e.g. CA, NY)
        
    Returns:
        str: Formatted string containing all active alerts for the state
    """
    # Construct the API URL for the specified state
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    # Handle various error cases
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    # Format each alert and join them with separators
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()  # Decorator to register this function as an MCP tool
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.
    
    This tool fetches a detailed weather forecast for a specific location
    using the National Weather Service API. It requires two steps:
    1. Get the forecast grid endpoint for the location
    2. Use that endpoint to get the actual forecast data
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        
    Returns:
        str: Formatted string containing the weather forecast
    """
    # Step 1: Get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Step 2: Get the actual forecast using the URL from step 1
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the forecast periods into a readable string
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods for brevity
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    # Initialize and run the server using stdio transport
    # This allows the server to communicate through standard input/output
    mcp.run(transport='stdio') 