import { useRef, useState } from "react";
import { MapContainer, TileLayer, FeatureGroup } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import L, { featureGroup } from "leaflet";

let DefaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});
L.Marker.prototype.options.icon = DefaultIcon;

function App() {
  const featureGroupRef = useRef<any>(null);
  const [coords, setCoords] = useState<any[]>([]);
  const [indices, setIndices] = useState<any>(null);
  const [healthStatus, setHealthStatus] = useState<string>("");
  const [healthyRange, setHealthyRange] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [cropType, setCropType] = useState<string>("wheat");

  const handleCreated = (e: any) => {
    const layer = e.layer;
    const polygon = layer.getLatLngs()[0].map((pt: any) => [pt.lng, pt.lat]);
    setCoords([polygon]);
  };

  const clearPolygons = () => {
    const layerGroup = featureGroupRef.current;
    if (layerGroup) {
      layerGroup.clearLayers();
      setCoords([]);
      setIndices(null);
      setHealthStatus("");
      setHealthyRange("");
    }
  };
  // ... previous imports and setup ...
    const analyzeFarm = async () => {
      if (!coords.length) return alert("Please draw a farm boundary first!");
      setLoading(true);
      try {
        const res = await fetch("http://127.0.0.1:5000/analyze-farm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ coordinates: coords, crop_type: cropType }),
        });

        const data = await res.json();
        if (data.error) {
          alert(data.error);
        } else {
          setIndices(data.indices);
          setHealthStatus(data.health_status);
          setHealthyRange(data.healthy_range); // take from backend

          const health = data.health_status || "Unknown";
          let color = "gray";
          if (health.includes("Healthy")) color = "green";
          else if (health.includes("Moderate")) color = "yellow";
          else if (health.includes("Unhealthy") || health.includes("excess")) color = "red";
          else if (health.includes("Bare")) color = "brown";

          featureGroupRef.current?.setStyle?.({ color, fillColor: color, fillOpacity: 0.3 });
        }
        
      } catch (err) {
        alert("Error connecting to backend!");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };


  return (
    <div style={{ padding: "1rem" }}>
      <h1>Gujarat AgriTech – Crop Health Analyzer</h1>

      <div style={{ marginBottom: "10px" }}>
        <label style={{ fontWeight: "bold", marginRight: "10px" }}>
          Select Crop Type:
        </label>
        <select
          value={cropType}
          onChange={(e) => setCropType(e.target.value)}
          style={{
            padding: "8px",
            borderRadius: "5px",
            border: "1px solid #ccc",
          }}
        >
          <option value="wheat">Wheat</option>
          <option value="maize">Maize</option>
          <option value="cotton">Cotton</option>
          <option value="rice">Rice</option>
          <option value="sugarcane">Sugarcane</option>
        </select>
      </div>

      <MapContainer
        center={[22.5645, 72.9289]}
        zoom={10}
        style={{ height: "500px", width: "100%", borderRadius: "10px" }}
      >
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="&copy; Esri"
        />
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
          opacity={0.3}
        />
        <FeatureGroup ref={featureGroupRef}>
          <EditControl
            position="topright"
            onCreated={(e: any) => {
              const layer = e.layer;
              const polygon = layer.getLatLngs()[0].map((pt: any) => [pt.lng, pt.lat]);
              console.log("Polygon:", polygon);
              setCoords([polygon]);

              // Save the layer for later style update
              if (featureGroupRef.current) {
                featureGroupRef.current.selectedLayer = layer;
              }
            }}
            draw={{
              polygon: true,
              rectangle: true,
              polyline: false,
              circle: false,
              marker: false,
              circlemarker: false,
            }}
            edit={{ remove: true}}
          />
        </FeatureGroup>
      </MapContainer>

      <div style={{ marginTop: "15px" }}>
        <button
          onClick={analyzeFarm}
          style={{
            padding: "10px 20px",
            background: "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            marginRight: "10px",
          }}
        >
          {loading ? "Analyzing..." : "Analyze Crop Health"}
        </button>

        <button
          onClick={clearPolygons}
          style={{
            padding: "10px 20px",
            background: "#f44336",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          Clear Selection
        </button>
      </div>

      {indices && (
        <div
          style={{
            marginTop: "25px",
            background: "#f6f6f6",
            padding: "15px",
            borderRadius: "8px",
            boxShadow: "0 0 5px rgba(0,0,0,0.1)",
          }}
        >
          <h2>Vegetation & Moisture Indices</h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "10px",
              marginTop: "10px",
            }}
          >
            {Object.entries(indices).map(([key, value]) => (
              <div
                key={key}
                style={{
                  background: "white",
                  padding: "10px",
                  borderRadius: "6px",
                  textAlign: "center",
                  boxShadow: "0 0 3px rgba(0,0,0,0.1)",
                }}
              >
                <h4>{key}</h4>
                <p style={{ fontSize: "1.2rem", fontWeight: "bold" }}>
                  {String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {healthyRange && (
        <div
          style={{
            marginTop: "15px",
            background: "#fffbe6",
            padding: "12px",
            borderRadius: "8px",
          }}
        >
          <strong>Healthy NDVI Range for {cropType}:</strong> {healthyRange}
        </div>
      )}


      {healthStatus && (
        <div
          style={{
            marginTop: "20px",
            background: "#fff3cd",
            padding: "15px",
            borderRadius: "8px",
          }}
        >
          <h3>Crop Health Status</h3>
          <p>{healthStatus}</p>
        </div>
      )}
    </div>
  );
}

export default App;
