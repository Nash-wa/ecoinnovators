import argparse
from pathlib import Path
import ee
import geemap

# --------------------------------------------------
# Initialize Google Earth Engine
# IMPORTANT:
# This will work ONLY after your project is registered
# for Earth Engine access in Google Cloud Console.
# --------------------------------------------------
ee.Initialize(project="ecoinnovators-480506")


def fetch_satellite_image(lat, lon, out_path, buffer_m=120):
    """
    Fetch a real satellite image using Sentinel-2 via Google Earth Engine.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create point and buffer region
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m).bounds()

    # Load Sentinel-2 imagery (UPDATED, not deprecated)
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2024-01-01", "2024-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
        .select(["B4", "B3", "B2"])  # RGB bands
    )

    # Export image to local file
    # NOTE: image is passed as FIRST ARGUMENT (no image= keyword)
    geemap.ee_export_image(
        image,
        filename=str(out_path),
        scale=10,
        region=region,
        file_per_band=False
    )

    print(f"[OK] Real satellite image saved to {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Lat/Lon → Satellite Image (Earth Engine)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--out", type=str, default="outputs/images/real.png")
    args = parser.parse_args()

    fetch_satellite_image(args.lat, args.lon, args.out)


if __name__ == "__main__":
    main()
