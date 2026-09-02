import requests
import os

# ====================== CONFIG ======================
API_KEY = os.environ["NREL_API_KEY"]
EMAIL = "cb.ps.i5das23032@cb.students.amrita.edu"            # ← Replace this

# Golden, Colorado (NREL campus - matches your PVDAQ systems)
LAT = 39.74
LON = -105.17

# Years you want (you can change this list)
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

# Variables to download
ATTRIBUTES = "ghi,dni,dhi,air_temperature,wind_speed,surface_pressure,relative_humidity"

# Output folder
OUTPUT_FOLDER = "nsrdb_golden"
# ====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for year in YEARS:
    print(f"Downloading year {year}...")

    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

    params = {
        "api_key": API_KEY,
        "wkt": f"POINT({LON} {LAT})",
        "names": str(year),
        "interval": "60",                    # 60-minute data (change to 30 if needed)
        "email": EMAIL,
        "attributes": ATTRIBUTES,
        "full_name": "Researcher",
        "affiliation": "University",
        "reason": "research",
        "utc": "false",
        "leap_day": "true"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        filename = os.path.join(OUTPUT_FOLDER, f"nsrdb_golden_{year}.csv")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✓ Saved: {filename}")
    else:
        print(f"✗ Failed for {year}. Status: {response.status_code}")
        print(response.text[:300])

print("\nDone!")