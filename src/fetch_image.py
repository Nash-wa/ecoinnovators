import os
import argparse
from pathlib import Path
import requests


def fetch_satellite_image(lat, lon, out_path, zoom=20, size=(640, 640), scale=2, api_key=None):
    """
    Fetch a satellite image from Google Static Maps API.
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY environment variable.")

    url = "https://maps.googleapis.com/maps/api/staticmap"

    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "maptype": "satellite",
        "size": f"{size[0]}x{size[1]}",
        "scale": scale,
        "key": api_key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(response.content)

    print(f"Image saved to {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--out", type=str, default="outputs/images/output.png")
    parser.add_argument("--zoom", type=int, default=20)
    args = parser.parse_args()

    fetch_satellite_image(args.lat, args.lon, args.out, zoom=args.zoom)


if __name__ == "__main__":
    main()

