#FROM ikewai/task-rf-pre-base
FROM conda/miniconda3

RUN conda install pandas geopandas numpy rasterio matplotlib
RUN conda install requests

WORKDIR /docker-build
COPY download_ndvi.py .
COPY imputate_historical_ndvi_maps.py .
COPY mosaic_ndvi_and_make_animation.py .

# Get Directory list
#RUN wget https://raw.githubusercontent.com/ikewai/hawaii_climate_products_container/main/preliminary/air_temp/daily/docs/currentDirs.txt

# Make directories and download scripts
#RUN /bin/python3 txt2mkdir_airtemp.py currentDirs.txt
#RUN /bin/python3 json2wget.py script_manifest.json

WORKDIR /sync

# Get pre-staged manifest of files to upload
#RUN wget https://raw.githubusercontent.com/ikewai/hawaii_climate_products_container/main/preliminary/air_temp/daily/docs/upload_config.json

# Copy json "upload" value inserter
#COPY ./upload_list_inserter.py upload_list_inserter.py

# Copy upload config template from build context
#COPY ./upload_config.json config.json

WORKDIR /actor
COPY task.sh .
RUN chmod +x task.sh