import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import parseGeoraster from "georaster";
import GeoRasterLayer from "georaster-layer-for-leaflet";

export default function GeotiffLayer({ frame, isVisible }) {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    let layer;

    async function loadTiff() {
      const response = await fetch(frame);
      const buffer = await response.arrayBuffer();
      const georaster = await parseGeoraster(buffer);

      layer = new GeoRasterLayer({
        georaster,
        opacity: isVisible ? 1 : 0,
        resolution: 64,
        pixelValuesToColorFn: (values) => {
        const [r, g, b, a] = values;
          // If the alpha channel is 0, return null to make it fully transparent
          if (a === 0) return null;
          // Map the 0-255 alpha value to the 0-1 range for CSS rgba
          return `rgba(${r},${g},${b},${a / 255})`;
        },
      });

      layer.addTo(map);
      layerRef.current = layer;
      map.fitBounds(layer.getBounds());
    }

    loadTiff();

    return () => {
      if (layerRef.current) map.removeLayer(layerRef.current);
    };
  }, [map, frame]);

  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.setOpacity(isVisible ? 1 : 0);
    }
  }, [isVisible]);

  return null;
}