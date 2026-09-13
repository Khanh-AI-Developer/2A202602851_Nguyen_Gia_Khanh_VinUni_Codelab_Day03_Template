import json
import os
from typing import List, Dict, Any

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

def get_flight_info(origin: str, destination: str, max_price: int = 5000000) -> List[Dict[str, Any]]:
    """
    Search for flights matching origin, destination, and budget constraint.
    """
    flight_file = os.path.join(RAW_DATA_DIR, "flight_data.json")
    if not os.path.exists(flight_file):
        return []

    if not isinstance(origin, str) or not isinstance(destination, str):
        return []
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        return []

    with open(flight_file, "r", encoding="utf-8") as f:
        flights = json.load(f)

    origin = origin.strip().upper()
    destination = destination.strip().upper()
    results = [
        fl for fl in flights
        if fl.get("origin", "").strip().upper() == origin
        and fl.get("destination", "").strip().upper() == destination
        and fl.get("price_vnd", float("inf")) <= max_price
    ]
    return results

def get_weather_forecast(city_code: str) -> Dict[str, Any]:
    """
    Get weather forecast and outfit recommendation for a city code (e.g. SGN, HAN, DAD).
    """
    weather_file = os.path.join(RAW_DATA_DIR, "weather_data.json")
    if not os.path.exists(weather_file):
        return {"error": "Weather data not found"}
    
    if not isinstance(city_code, str) or not city_code.strip():
        return {"error": "City code is required"}

    with open(weather_file, "r", encoding="utf-8") as f:
        weather_data = json.load(f)

    city_code = city_code.strip().upper()
    return weather_data.get(city_code, {"error": f"No data for {city_code}"})

# Tool Registry for ReAct Agent
TOOL_DEFINITIONS = [
    {
        "name": "get_flight_info",
        "description": "Tìm chuyến bay theo điểm đi, điểm đến và giá tối đa.",
        "parameters": {
            "origin": "Mã sân bay đi (VD: HAN)",
            "destination": "Mã sân bay đến (VD: SGN)",
            "max_price": "Giá vé tối đa dạng số nguyên (VND)"
        }
    },
    {
        "name": "get_weather_forecast",
        "description": "Lấy thông tin thời tiết và gợi ý trang phục theo mã sân bay/thành phố (SGN, HAN, DAD).",
        "parameters": {
            "city_code": "Mã sân bay thành phố (VD: SGN)"
        }
    }
]

TOOL_MAP = {
    "get_flight_info": get_flight_info,
    "get_weather_forecast": get_weather_forecast
}
