### GIS converter tool: projectRadarData.py
This script converts the radar reflectivity channel in the .npz detection arrays (files produced in nfgda_detection/) to a georeferenced file that can be displayed in a GIS. This converter script requires GDAL. To use the GDAL python bindings, you first need to install GDAL on your system. 

* Ubuntu 24.04 - GDAL 3.12.1 is provided by the apt repo:
````
sudo apt install gdal
````
* Ubuntu 22.04 - The GDAL version (3.4.1) provided by the apt repo is not compatible with Python 3.12 (required by the NFGDA venv). Instead, a GDAL version 3.12.1 must be compiled from source:

````
sudo apt update && sudo apt upgrade
sudo apt install build-essential cmake libproj-dev libgdal-dev gdal-bin python3-gdal
wget https://github.com/OSGeo/gdal/releases/download/v3.12.1/gdal-3.12.1.tar.gz
tar -xvf gdal-3.12.1.tar.gz
cd gdal-3.12.1
mkdir build
cd build
cmake ..
cmake --build .
sudo cmake --build . --target install
sudo ldconfg
````
Now install the python gdal package into the NFGDA venv:
````
pip install gdal
````
Example usage:
````
./projectRadarData.py nf_predKABX20200707_012805_V06.npz 35.149722 -106.823889
````

