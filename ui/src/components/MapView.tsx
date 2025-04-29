import "leaflet/dist/leaflet.css";
import L, { Icon } from "leaflet";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Tooltip,
  CircleMarker,
} from "react-leaflet";

import AisShipSymbol from "./AisShipSymbol";
import { Loader } from "lucide-react";

type props = { data: any; showTracked: boolean };

const hydroIcon = new Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/156/156318.png",
  iconSize: [16, 16],
});

const trackedIcon = new Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/592/592255.png",
  iconSize: [20, 20],
});

const getLastElem = (arr: any[]): any => {
  const len = arr.length;
  return len > 0 ? arr[len - 1] : 0;
};

export default function MapView({ data, showTracked }: props) {
  if (!data)
    return (
      <div className="absolute left-1/2 top-1/2 text-primary">
        <Loader className="animate-spin" />
      </div>
    );

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
          >
            <Tooltip className="flex flex-col">
              <p className="font-bold">HYDRO {hydro.id}</p>
              <p>
                Expected Pressure{" "}
                <span className="font-semibold">
                  {hydro.expected_pressure}dB
                </span>
              </p>
              <p>
                Observed Pressure{" "}
                <span className="font-semibold">
                  {getLastElem(hydro.observed_pressure)}dB
                </span>
              </p>
              <p>
                Delta{" "}
                <span className="font-semibold">
                  {Number(getLastElem(hydro.observed_pressure)) -
                    Number(hydro.expected_pressure)}
                </span>
              </p>
            </Tooltip>
          </Marker>
        );
      })}
      {showTracked
        ? Object.entries(data.tracking).map(([key, value], index) => {
            return (
              <Marker key={index * 100} icon={trackedIcon} position={value}>
                <Tooltip className="flex flex-col">
                  <p>{key}</p>
                  <p>
                    position <span className="font-bold">{value}</span>
                  </p>
                </Tooltip>
              </Marker>
            );
          })
        : null}
    </MapContainer>
  );
}
