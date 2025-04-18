import "leaflet/dist/leaflet.css";
import L, { Icon } from "leaflet";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  CircleMarker,
} from "react-leaflet";

import AisShipSymbol from "./AisShipSymbol";

type props = { data: any };

const hydroIcon = new Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/156/156318.png",
  iconSize: [16, 16],
});

export default function MapView({ data }: props) {
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
      />
      {ships.map((ship, index: number) => {
        return (
          <AisShipSymbol
            key={index}
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
            key={-index}
            icon={hydroIcon}
            position={[hydro.latitude, hydro.longitude]}
          ></Marker>
        );
      })}
    </MapContainer>
  );
}
