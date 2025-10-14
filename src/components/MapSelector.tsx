import { useState } from "react";
import { MapContainer, TileLayer, useMapEvents, Polygon } from "react-leaflet";
import "leaflet/dist/leaflet.css";

interface Props {
  onAreaSelected: (coords: number[][]) => void;
}

const MapSelector: React.FC<Props> = ({ onAreaSelected }) => {
  const [coords, setCoords] = useState<number[][]>([]);

  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      const newCoords = [...coords, [lat, lng]];
      setCoords(newCoords);
      onAreaSelected(newCoords);
    },
  });

  return (
    <MapContainer
      center={[22.5645, 72.9289]} // Example: Gujarat
      zoom={10}
      style={{ height: "500px", width: "100%" }}
    >
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
      />


      {coords.length > 2 && (
        <Polygon positions={coords as [number, number][]} color="green" />
      )}
    </MapContainer>
  );
};

export default MapSelector;
