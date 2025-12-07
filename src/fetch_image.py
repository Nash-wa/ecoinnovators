import os
import argparse
from pathlib import Path
import requests
import shutil


# CHANGE THIS to the real image path in your repo!
FALLBACK_LOCAL_IMAGE = r"data/raw/dataset1/test/tile_z18_x183116_y104722_jpg.rf.6948dd2c4e49507364f034be29e02ded.jpg"


def fetch_satellite_image(lat, lon, out_path, zoom=20, size=(640, 640), scale=2, api_key=None):
    """
    Try Google Static Maps API. If it fails, use fallback local image.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get API key from environment
    if api_key is None:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")

    # If no key → fallback immediately
    if not api_key:
        print("[WARN] No API key found. Using fallback image.")
        return use_fallback_image(out_path)

    # Google API call
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "maptype": "satellite",
        "size": f"{size[0]}x{size[1]}",
        "scale": scale,
        "key": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # will throw 403 error here
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"[OK] Downloaded Google image to {out_path}")
        return str(out_path)

    except Exception as e:
        print(f"[WARN] Google fetch failed ({e}). Using fallback instead.")
        return use_fallback_image(out_path)


def use_fallbac_
