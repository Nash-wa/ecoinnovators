# debug_fetch_image.py (overwrite src\fetch_image.py with this)
import os
import argparse
from pathlib import Path
import requests
import shutil
import sys
import textwrap

# Fallback path - (your repo path). Keep as the relative path you gave earlier.
FALLBACK_LOCAL_IMAGE = r"data/raw/dataset1/test/tile_z18_x183116_y104722_jpg.rf.6948dd2c4e49507364f034be29e02ded.jpg"

def debug_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def fetch_satellite_image(lat, lon, out_path, zoom=20, size=(640, 640), scale=2, api_key=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_print("== DEBUG START ==")
    debug_print("Out path:", out_path)

    # 1) show which file we are executing
    debug_print("Script file:", Path(__file__).resolve())

    # 2) API key sourcing
    if api_key is None:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    debug_print("GOOGLE_MAPS_API_KEY (env var):", "<SET>" if api_key else "<NOT SET>")

    # 3) fallback exists?
    fallback = Path(FALLBACK_LOCAL_IMAGE)
    debug_print("Fallback path:", fallback)
    debug_print("Fallback exists:", fallback.exists())

    if not api_key:
        debug_print("[INFO] No API key found. Will use fallback if available.")
        return use_fallback_image(out_path)

    # 4) Build URL and print it (without key)
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "maptype": "satellite",
        "size": f"{size[0]}x{size[1]}",
        "scale": scale,
        "key": api_key,
    }
    # Print the request (mask key)
    params_no_key = params.copy()
    params_no_key["key"] = "<HIDDEN_KEY>"
    debug_print("Request params:", params_no_key)

    try:
        debug_print("[INFO] Sending request to Google Static Maps...")
        resp = requests.get(url, params=params, timeout=15)
        debug_print("HTTP status code:", resp.status_code)
        # print a small part of response for diagnosing
        text_snippet = (resp.text[:400] + "...") if resp.text else "<no text>"
        debug_print("Response text snippet:", text_snippet)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        debug_print("[OK] Downloaded Google image to", out_path)
        debug_print("== DEBUG END ==")
        return str(out_path)

    except Exception as e:
        debug_print(f"[WARN] Google fetch failed: {repr(e)}")
        debug_print("[INFO] Falling back to local image.")
        return use_fallback_image(out_path)


def use_fallback_image(out_path: Path) -> str:
    fallback = Path(FALLBACK_LOCAL_IMAGE)
    if not fallback.exists():
        debug_print("[ERROR] Fallback image missing at:", fallback)
        raise FileNotFoundError(
            f"Fallback image not found at {fallback}. Update FALLBACK_LOCAL_IMAGE to a valid file."
        )
    shutil.copy(fallback, out_path)
    debug_print("[OK] Copied fallback image to", out_path)
    debug_print("== DEBUG END ==")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Debug fetch image (with fallback).")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--out", type=str, default="outputs/images/output.png")
    parser.add_argument("--zoom", type=int, default=20)
    args = parser.parse_args()

    try:
        fetch_satellite_image(args.lat, args.lon, args.out, zoom=args.zoom)
    except Exception as e:
        debug_print("[FATAL] Exception in main:", repr(e))
        debug_print(textwrap.dedent("""
            HELP:
            - Check FALLBACK_LOCAL_IMAGE path and ensure file exists inside repo.
            - Check that you saved this file and are running the correct file.
            - If API fetch fails and fallback missing, the script will raise.
            """))
        raise


if __name__ == "__main__":
    main()
