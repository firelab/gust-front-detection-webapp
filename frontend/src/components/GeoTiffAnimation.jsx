import { useEffect, useState } from "react";
import { ImageOverlay } from "react-leaflet";
import L from "leaflet";
import parseGeoraster from "georaster";

export default function GeotiffAnimation({ frames, currentIndex }) {
  const [frameData, setFrameData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function processFrames() {
      const processed = await Promise.all(frames.map(async (url) => {
        const res = await fetch(url);
        const buf = await res.arrayBuffer();
        const geo = await parseGeoraster(buf);

        const canvas = document.createElement('canvas');
        const w = canvas.width = geo.width;
        const h = canvas.height = geo.height;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(w, h);
        const data = imgData.data;

        // Cache band references
        const bands = geo.values;
        const numBands = bands.length;

        for (let i = 0; i < w * h; i++) {
          const x = i % w;
          const y = (i / w) | 0;
          const i4 = i << 2;

          if (numBands >= 3) {
            data[i4]     = bands[0][y][x]; // R
            data[i4 + 1] = bands[1][y][x]; // G
            data[i4 + 2] = bands[2][y][x]; // B
            data[i4 + 3] = numBands === 4 ? bands[3][y][x] : 255;
          } else {
            const v = bands[0][y][x];
            data[i4] = data[i4 + 1] = data[i4 + 2] = v;
            data[i4 + 3] = 255;
          }
        }
        ctx.putImageData(imgData, 0, 0);

        const southWest = L.Projection.SphericalMercator.unproject(L.point(geo.xmin, geo.ymin));
        const northEast = L.Projection.SphericalMercator.unproject(L.point(geo.xmax, geo.ymax));

        return {
          url: canvas.toDataURL(),
          bounds: [
            [southWest.lat, southWest.lng],
            [northEast.lat, northEast.lng]
          ]
        };
      }));

      setFrameData(processed);
      setLoading(false);
    }
    if (frames?.length) processFrames();
  }, [frames]);

  if (loading || !frameData[currentIndex]) return null;

  return (
    <ImageOverlay
      url={frameData[currentIndex].url}
      bounds={frameData[currentIndex].bounds}
      zIndex={1000}
      opacity={0.7}
    />
  );
}