import { useEffect } from "react";
import { useMap } from "react-leaflet";
import parseGeoraster from "georaster";
import GeoRasterLayer from "georaster-layer-for-leaflet";

export default function GeotiffLayer({ frame }) {
  const map = useMap();
  console.log("rendering frame: " + frame);

  useEffect(() => {
    let layer;

    async function loadTiff() {
      const response = await fetch(frame);
      const buffer = await response.arrayBuffer();
      const georaster = await parseGeoraster(buffer);

      layer = new GeoRasterLayer({
        georaster,
        opacity: 1,
        resolution: 256,
        pixelValuesToColorFn: (values) => {
        const [r, g, b, a] = values;
          // If the alpha channel is 0, return null to make it fully transparent
          if (a === 0) return null;
          // Map the 0-255 alpha value to the 0-1 range for CSS rgba
          return `rgba(${r},${g},${b},${a / 255})`;
        },
      });

      layer.addTo(map);
      map.fitBounds(layer.getBounds());
    }

    loadTiff();

    return () => {
      if (layer) map.removeLayer(layer);
    };
  }, [map, frame]);

  return null;
}