//import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet';
import { useEffect, useState } from 'react';
import axios from 'axios';

const REFRESH_INTERVAL = 1000;

export default function MapView() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get('http://localhost:8000/api/data');
        setData(res.data);
      } catch (err) {
        console.error("Data fetch failed", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  if (!data) return <p>Caricamento dati...</p>;
  console.log(data)
  return <p>loaded</p>
/*
  return (
    <MapContainer
      center={data.}
      zoom={5}
      style={{ height: '100vh', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {data.ships.map((ship) => (
        <CircleMarker
          key={ship.id}
          center={[ship.latitude, ship.longitude]}
          radius={10}
          pathOptions={{ color: ship.is_dark ? 'red' : 'green' }}
        >
          <Popup>
            Nave {ship.id} <br />
            {ship.is_dark ? "Dark" : "AIS"}
          </Popup>
        </CircleMarker>
      ))}

      {data.hydrophones.map((h) => (
        <Marker
          key={h.id}
          position={[h.latitude, h.longitude]}
        >
          <Popup>
            Idrofono {h.id}
            <br />
            Pressione: {h.observed_pressure.toFixed(2)}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
        */
}

