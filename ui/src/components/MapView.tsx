import "leaflet/dist/leaflet.css";
import L, { Icon } from "leaflet";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  CircleMarker,
} from "react-leaflet";

import { useEffect, useState } from "react";
import axios from "axios";

import { status } from "../App";
import AisShipSymbol from "./AisShipSymbol";

const REFRESH_INTERVAL = 1000;

type props = { status: status };

const hydroIcon = new Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/156/156318.png",
  iconSize: [16, 16],
});

export default function MapView({ status }: props) {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:8000/api/data");
        setData(res.data);
      } catch (err) {
        console.error("Data fetch failed", err);
      }
    };

    let interval: number = 0;
    fetchData();
    interval = setInterval(fetchData, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, [status]);

  if (!data) return <div>Loading...</div>;

  const area = data.area;
  const ships = data.ships ?? [];
  const hydrophones = data.hydrophones ?? [];

  return (
    <MapContainer
      center={[(area[0] + area[1]) / 2, (area[2] + area[3]) / 2]}
      zoom={7}
      scrollWheelZoom={false}
      className="h-full"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        //url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
      />
      {ships.map((ship, index: number) => {
        return (
          <AisShipSymbol
            key={ship.id}
            id={ship.id.toString()}
            position={[ship.latitude, ship.longitude]}
            heading={ship.heading ?? 0}
            speed={ship.speed ?? 0}
            isDark={ship.is_dark}
          />
        );
      })}
      {hydrophones.map((hydro, index: number) => {
        return (
          <Marker
            icon={hydroIcon}
            position={[hydro.latitude, hydro.longitude]}
          ></Marker>
        );
      })}
    </MapContainer>
  );
}
