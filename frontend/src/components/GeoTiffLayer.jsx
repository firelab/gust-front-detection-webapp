import { useEffect } from "react";
import { useMap } from "react-leaflet";
import parseGeoraster from "georaster";
import GeoRasterLayer from "georaster-layer-for-leaflet";

export default function GeotiffLayer() {
  const map = useMap();

  useEffect(() => {
    let layer;

    async function loadTiff() {
      const response = await fetch("/radar_reflectivity_3857.tif"); // currently this is a hardcoded file in frontend/public
      const buffer = await response.arrayBuffer();
      const georaster = await parseGeoraster(buffer);

      layer = new GeoRasterLayer({
        georaster,
        opacity: 0.7,
        resolution: 256,
      });

      layer.addTo(map);
      map.fitBounds(layer.getBounds());
    }

    loadTiff();

    return () => {
      if (layer) map.removeLayer(layer);
    };
  }, [map]);

  return null;
}
