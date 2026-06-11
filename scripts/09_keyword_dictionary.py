import pandas as pd
from pathlib import Path


OUTPUT_FOLDER = Path("KEYWORDS")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ── Keyword dictionaries ──────────

# ── Robot vacuum category ────────────────────────────────────────────────────
# Subreddits: r/RobotVacuums, r/Roborock, r/roomba

robot_vacuum_keywords = [
    # Features
    "suction", "mapping", "navigation", "lidar", "laser", "camera",
    "obstacle", "avoidance", "dock", "docking", "charge", "charging",
    "battery", "runtime", "schedule", "scheduling", "zone", "no-go",
    "virtual wall", "boundary", "mop", "mopping", "sweep", "sweeping",
    "filter", "hepa", "brush", "roller", "bin", "dustbin", "empty",
    "auto-empty", "self-empty", "noise", "quiet", "loud", "carpet",
    "hardwood", "floor", "threshold", "cliff", "sensor", "wifi",
    "app", "alexa", "google home", "homekit", "smart home", "map",
    "room", "multi-floor", "multi-level", "firmware", "update",
    "warranty", "support", "customer service", "return",

    # Performance
    "suction power", "pa", "pickup", "clean", "dirt", "debris",
    "hair", "pet hair", "fur", "dust", "allergen",

    # Brand names and models
    "roomba", "irobot", "roborock", "eufy", "dreame", "ecovacs",
    "deebot", "neato", "botvac", "shark", "ion", "braava",
    "s series", "q series", "p series", "e series", "j series",
    "s5", "s6", "s7", "s8", "q5", "q7", "q8",
    "980", "960", "i7", "i3", "j7", "s9",
]

# ── Kitchen appliance category ───────────────────────────────────────────────
# Subreddits: r/airfryer, r/instantpot

kitchen_keywords = [
    # Air fryer features
    "air fry", "airfry", "basket", "tray", "rack", "preheat",
    "crispy", "crunchy", "temperature", "temp", "degrees", "fahrenheit",
    "celsius", "timer", "wattage", "watts", "capacity", "quart",
    "litre", "liter", "drawer", "dual basket", "dual zone",

    # Instant pot / pressure cooker features
    "pressure cook", "pressure cooker", "instant pot", "instapot",
    "slow cook", "slow cooker", "saute", "sauté", "steam", "steaming",
    "rice", "yogurt", "yoghurt", "seal", "sealing", "valve",
    "vent", "venting", "float valve", "burn notice", "psi",
    "natural release", "quick release", "manual release",

    # Shared kitchen features
    "nonstick", "non-stick", "coating", "teflon", "ceramic",
    "dishwasher", "safe", "clean", "cleaning", "accessory",
    "accessories", "recipe", "cook", "cooking", "food", "meal",
    "preheat", "overheat", "smoke", "smell", "odor",
    "warranty", "support", "customer service", "return", "defect",
    "noise", "loud", "quiet",

    # Brand names and models
    "ninja", "cosori", "philips", "instant pot", "breville",
    "powerxl", "power xl", "gourmia", "chefman", "dash",
    "emeril", "nuwave", "vortex", "foodi", "duo", "pro",
    "ultra", "max", "xl", "deluxe", "smart",
    "duo nova", "duo plus", "duo crisp", "air fryer lid",
]

# ── Save to CSV for review ───────────────────────────────────────────────────

df_robot = pd.DataFrame({
    "keyword":  robot_vacuum_keywords,
    "category": "robot_vacuum",
    "subreddits": "RobotVacuums, Roborock, roomba"
})

df_kitchen = pd.DataFrame({
    "keyword":  kitchen_keywords,
    "category": "kitchen_appliance",
    "subreddits": "airfryer, instantpot"
})

df_robot.to_csv(OUTPUT_FOLDER / "keywords_robot_vacuum.csv", index=False)
df_kitchen.to_csv(OUTPUT_FOLDER / "keywords_kitchen_appliance.csv", index=False)

print("=" * 60)
print("Keyword dictionaries saved for review")
print("=" * 60)
print(f"\n  keywords_robot_vacuum.csv    — {len(df_robot)} terms")
print(f"  keywords_kitchen_appliance.csv — {len(df_kitchen)} terms")
print(f"\n  Saved to: {OUTPUT_FOLDER.resolve()}")


