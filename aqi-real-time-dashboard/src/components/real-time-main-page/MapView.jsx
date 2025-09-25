import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function ChangeMapView({ center }) {
  const map = useMap();
  map.setView(center, 10);
  return null;
}

const MapView = ({ selectedStation, currentAQI }) => {
  return (
    <div className="map-container">
      <MapContainer
        center={[selectedStation.latitude, selectedStation.longitude]}
        zoom={10}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Marker position={[selectedStation.latitude, selectedStation.longitude]}>
          <Popup>
            {selectedStation.name}, {selectedStation.city || 'Unknown'}
            <br />
            AQI: {currentAQI !== null ? Math.round(currentAQI) : 'Loading...'}
          </Popup>
        </Marker>
        <ChangeMapView center={[selectedStation.latitude, selectedStation.longitude]} />
      </MapContainer>

     
    </div>
    
  );
};

export default MapView;
