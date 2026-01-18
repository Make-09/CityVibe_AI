import numpy as np
import os
from dotenv import load_dotenv
from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection, BBox, 
    CRS, bbox_to_dimensions, MimeType
)

load_dotenv("api_keys.env")

config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

def get_ndvi_multi_radius(lat: float, lon: float, radii=[100, 200, 300]):
    results = {}
    
    for radius in radii:
        delta = radius / 111000
        bbox = BBox([lon - delta, lat - delta, lon + delta, lat + delta], crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=10)

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08", "dataMask"],
                output: { bands: 1, sampleType: "FLOAT32" }
            };
        }
        function evaluatePixel(sample) {
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            return (sample.dataMask === 1) ? [ndvi] : [NaN];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2024-06-01", "2024-09-01"),
                mosaicking_order="mostRecent" 
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config
        )

        data = request.get_data()[0].flatten()
        data = data[~np.isnan(data)]
        
        if data.size > 0:
            data_sorted = np.sort(data)
            top_30_idx = int(len(data_sorted) * 0.7)
            val = np.mean(data_sorted[top_30_idx:])
            results[radius] = float(val)
        else:
            results[radius] = 0.0
            
    return results

if __name__ == "__main__":
    lat, lon = 51.27361102351295, 51.42923776746755
    analysis = get_ndvi_multi_radius(lat, lon)
    
    print("--- АУДИТ CITYVIBE AI ---")
    for r, val in analysis.items():
        print(f"Радиус {r}м: NDVI = {val:.4f}")