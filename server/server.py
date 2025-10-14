from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import requests
import json
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()
google_project_id = os.getenv("GOOGLE_PROJECT_ID")

# Initialize Google Earth Engine
ee.Initialize(project=google_project_id)

# Load crop NDVI ranges from JSON
with open("crop_ndvi_ranges.json") as f:
    crop_ranges = json.load(f)

@app.route("/analyze-farm", methods=["POST"])
def analyze_farm():
    try:
        data = request.get_json()
        coords = data.get("coordinates")
        crop_type = data.get("crop_type", "").lower()

        if not coords:
            return jsonify({"error": "No coordinates provided"}), 400
        if not crop_type:
            return jsonify({"error": "No crop type provided"}), 400

        ndvi_min = crop_ranges.get(crop_type, {}).get("ndvi_min")
        ndvi_max = crop_ranges.get(crop_type, {}).get("ndvi_max")

        # Create EE geometry
        polygon = ee.Geometry.Polygon(coords)

        # Sentinel-2 ImageCollection
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(polygon)
            .filterDate("2025-08-01", "2025-10-11")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )

        if collection.size().getInfo() == 0:
            return jsonify({"error": "No Sentinel-2 images found for the given area/date range."}), 404

        image = collection.median()

        # Required bands
        bands = ["B2", "B3", "B4", "B8", "B11"]
        available_bands = image.bandNames().getInfo()
        missing = [b for b in bands if b not in available_bands]
        if missing:
            return jsonify({"error": f"Missing bands: {missing}. Available: {available_bands}"}), 500

        # Select key bands
        blue = image.select("B2")
        red = image.select("B4")
        nir = image.select("B8")
        swir1 = image.select("B11")

        # Calculate indices
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
            {"NIR": nir, "RED": red, "BLUE": blue}
        ).rename("EVI")
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        msi = image.expression("SWIR / NIR", {"SWIR": swir1, "NIR": nir}).rename("MSI")

        # Combine indices
        indices_image = image.addBands([ndvi, evi, ndwi, msi])

        mean_stats = indices_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=polygon,
            scale=10,
            maxPixels=1e13
        ).getInfo()

        def safe_val(name):
            val = mean_stats.get(name)
            return round(val, 4) if val is not None else 0.0

        # Weather info
        centroid = polygon.centroid().coordinates().getInfo()
        lon, lat = centroid
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current="
            "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m"
        )
        weather_res = requests.get(weather_url).json()
        current = weather_res.get("current", {})

        temp = current.get("temperature_2m", 0)
        rain = current.get("rain", 0)
        mean_ndvi = safe_val("NDVI")

        # Crop health assessment based on NDVI ranges
        if mean_ndvi < ndvi_min:
            health_status = f"{crop_type.capitalize()} shows stress or poor vegetation (Unhealthy). OR maybe bare soil."
        elif ndvi_min <= mean_ndvi <= ndvi_max:
            health_status = f"{crop_type.capitalize()} crop appears Healthy."
        else:
            health_status = f"Excess greenness detected (possibly weeds or dense canopy)."

        return jsonify({
            "indices": {
                "NDVI": safe_val("NDVI"),
                "EVI": safe_val("EVI"),
                "NDWI": safe_val("NDWI"),
                "MSI": safe_val("MSI"),
                "GREEN": safe_val("B3"),
                "RED": safe_val("B4"),
                "NIR": safe_val("B8"),
            },
            "weather": {
                "temperature": temp,
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "rain": rain,
            },
            "crop_type": crop_type,
            "healthy_range": f"{ndvi_min} - {ndvi_max}",
            "health_status": health_status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
