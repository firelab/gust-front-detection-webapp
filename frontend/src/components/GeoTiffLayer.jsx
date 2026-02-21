import { useEffect } from "react";
import { useMap } from "react-leaflet";
import parseGeoraster from "georaster";
import GeoRasterLayer from "georaster-layer-for-leaflet";

// Linear color interpolation helper
function lerpColor(a, b, t) {
  const ah = parseInt(a.replace("#", ""), 16);
  const bh = parseInt(b.replace("#", ""), 16);
  const ar = (ah >> 16) & 255, ag = (ah >> 8) & 255, ab = ah & 255;
  const br = (bh >> 16) & 255, bg = (bh >> 8) & 255, bb = bh & 255;
  const rr = Math.round(ar + t * (br - ar));
  const rg = Math.round(ag + t * (bg - ag));
  const rb = Math.round(ab + t * (bb - ab));
  return `rgb(${rr},${rg},${rb})`;
}

export default function GeotiffLayer() {
  const map = useMap();

  useEffect(() => {
    let layer;

    async function loadTiff() {
      const response = await fetch("/radar_reflectivity_3857.tif"); // currently this is a hardcoded file in frontend/public
      const buffer = await response.arrayBuffer();
      const georaster = await parseGeoraster(buffer);
      const MIN = 0;
      const MAX = 35; // This may need to be adjusted depending on data

      layer = new GeoRasterLayer({
        georaster,
        opacity: 0.7,
        resolution: 256,

        pixelValuesToColorFn: (values) => {
          const v = values[0];
          if (v == null || Number.isNaN(v)) return null;
          const t = Math.max(0, Math.min(1, (v - MIN) / (MAX - MIN))); // Normalize value to 0–1
          //cyan -> blue -> green -> yellow -> red
          if (t < 0.10) return lerpColor("#00ffff", "#0000ff", t / 0.25);
          if (t < 0.5) return lerpColor("#0000ff", "#00ff00", (t - 0.25) / 0.25);
          if (t < 0.75) return lerpColor("#00ff00", "#ffff00", (t - 0.5) / 0.25);
          return lerpColor("#ffff00", "#ff0000", (t - 0.75) / 0.25);
        },
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