import { useMap } from "react-leaflet";
import { useEffect, useRef } from "react";
import TrackSymbol from "@arl/leaflet-tracksymbol2";

type Props = {
  id: string;
  position: [number, number];
  heading?: number;
  speed?: number;
  isDark?: boolean;
};

const degToRads = (deg) => (deg * Math.PI) / 180.0;

export default function AisShipSymbol({
  id,
  position,
  heading = 0,
  speed = 0,
  isDark = false,
}: Props) {
  const map = useMap();
  const shipRef = useRef<any>(null);

  useEffect(() => {
    const ship = new TrackSymbol(position, {
      fill: true,
      fillColor: isDark ? "red" : "yellow",
      fillOpacity: 1,
      speed: speed,
      heading: degToRads(heading),
      course: degToRads(heading),
    });
    ship.bindTooltip("TrackSymbol1");
    ship.addTo(map);

    ship.addTo(map);
    shipRef.current = ship;

    return () => {
      map.removeLayer(ship);
    };
  }, [map]);

  // Aggiorna posizione, heading, sog se cambiano
  useEffect(() => {
    if (shipRef.current) {
      shipRef.current.setLatLng(position);
      shipRef.current.setHeading(degToRads(heading));
      shipRef.current.setSpeed(speed);
      shipRef.current.setCourse(degToRads(heading));
    }
  }, [position, heading, speed]);

  return null;
}
