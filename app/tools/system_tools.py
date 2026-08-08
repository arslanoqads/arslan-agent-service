import random
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# --- TOOL 1: RAM Usage (simulated for Cloud Run) ---
class RAMUsageInput(BaseModel):
    pass


@tool("get_ram_usage", args_schema=RAMUsageInput)
def get_ram_usage() -> str:
    """Returns simulated system memory/RAM usage metrics."""
    total_gb = 16.0
    percent = round(random.uniform(35.0, 82.0), 1)
    available_gb = round(total_gb * (100 - percent) / 100, 2)
    return (
        f"[SIMULATED] RAM Usage: {percent}% used "
        f"(Available: {available_gb} GB / Total: {total_gb:.2f} GB)"
    )


# --- TOOL 2: Battery Status (simulated for Cloud Run) ---
class BatteryInput(BaseModel):
    pass


@tool("get_battery_status", args_schema=BatteryInput)
def get_battery_status() -> str:
    """Returns simulated device battery percentage and charging status."""
    percent = random.randint(15, 100)
    status = random.choice(["Plugged in", "Discharging"])
    return f"[SIMULATED] Battery: {percent}% ({status})"


# --- TOOL 3: Create Temp File (simulated for Cloud Run) ---
class TempFileInput(BaseModel):
    content_note: str = Field(description="The note or message to write inside the temp file.")


@tool("create_timestamp_temp_file", args_schema=TempFileInput)
def create_timestamp_temp_file(content_note: str) -> str:
    """Simulates creating a temporary file with a timestamp and note."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_id = random.randint(100000, 999999)
    file_path = f"/tmp/agent_note_{file_id}.txt"
    return (
        f"[SIMULATED] Temp file created at: {file_path} "
        f"(Timestamp: {now_str}; Note: '{content_note}')"
    )
