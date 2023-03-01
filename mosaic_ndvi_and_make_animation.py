#############MOSAIC
import rasterio as rio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

# Import all MODIS chl-a Geotiff files
#base_dir = r'E:\KIEU TRANG\A_PHD\NDVI'
base_dir = r'D:\project\ndvi_hawaii\gap_filled ndvi data\hawaii state'

fnames = [os.path.join(base_dir, fname) for fname in sorted(os.listdir(base_dir))]

dfs_names = sorted(os.listdir(base_dir))

#fnames[0][124:131]

dfs_names[0][-11:-4]


#k=0
for i in range(len (fnames)):
    #i=k
    #Let’s first create an empty list for the datafiles that will be part of the mosaic.
    src_files_to_mosaic = []
    # print (i)
    #year_day=fnames[i][59:66]
    year_day=dfs_names[i][-11:-4]
    for j in range (len(fnames)):
        # print(j)
        # print(fnames [j])
        if int(dfs_names[j][-11:-4])==int(year_day):
            src = rio.open(fnames [j])
            src_files_to_mosaic.append(src)
    
    # It is important to have 8 images for each day otherwise mosaic will no be done correctly
    if len (src_files_to_mosaic)< 8:
        print ('error in number of images')
        break
    
    print(len(src_files_to_mosaic))
    #k=j+1
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)
    #show(mosaic, cmap='terrain')
    
    # Now we are ready to save our mosaic to disk
    # Copy the metadata
    out_meta = src.meta.copy()
    
    # Update the metadata
    
    out_meta.update({"driver": "GTiff","height": mosaic.shape[1],"width": mosaic.shape[2],"transform": out_trans})
    
    # Write the mosaic raster to disk
    # with rio.open(out_fp, "w", **out_meta) as dest: dest.write(mosaic)
    
    with rio.open(r'D:/project/ndvi_hawaii/gap_filled ndvi data/hawaii state_mosaic/NDVI_16day_250m_%s.tif'%int(year_day), 'w', **out_meta) as dst:
        dst.write(mosaic) 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rasterio as rio
#print(rio.__version__)
import os, shutil
# Draw first map to check it
base_dir = r'D:/project/ndvi_hawaii/gap_filled ndvi data/hawaii state_mosaic'

fnames = [os.path.join(base_dir, fname) for fname in sorted(os.listdir(base_dir))]

dfs_names = sorted(os.listdir(base_dir))
date=dfs_names[0][-11:-4]
from datetime import datetime, timedelta
image_date= datetime(int(date[0:4]),1,1)+timedelta(days=int(date[4:7]))
image_date
day_16 = image_date + timedelta(days=16)
print(image_date.year,'-',image_date.month,'-',image_date.day)
MODIS_ndvi_ini = rio.open(fnames[0], "r")


MODIS_ndvi_ini_meta = MODIS_ndvi_ini.profile
MODIS_ndvi_ini_pixelSizeX, MODIS_ndvi_ini_pixelSizeY = MODIS_ndvi_ini.res

MODIS_ndvi=MODIS_ndvi_ini.read(1)

bbox=MODIS_ndvi_ini.bounds
bbox
latitude=(bbox[1],bbox[3])
longitude=(bbox[0],bbox[2])
bbox = ((bbox[0],   bbox[2],      
         bbox[1], bbox[3]))
def data_solve_nan_ref(data):
    df_dem=pd.DataFrame (data)
    #df_dem_replace=df_dem.replace(32767.0,np.nan)
    df_dem_replace=df_dem.replace(123456,np.nan)
    return df_dem_replace.values

MODIS_ndvi=data_solve_nan_ref(MODIS_ndvi)    

fig, ax = plt.subplots(figsize=(10,15))
fig.patch.set_alpha(0)

plt.imshow(MODIS_ndvi, extent=bbox,cmap='RdYlGn',zorder=1)
im_ratio = MODIS_ndvi.shape[0]/MODIS_ndvi.shape[1]
plt.colorbar(fraction=0.047*im_ratio,label='ndvi')
#plt.colorbar(fraction=0.05*im_ratio,label='ndvi')
plt.grid(zorder=0)
plt.title('16 day composite ndvi from ' +str(image_date.year)+'/'+str(image_date.month)+'/'+str(image_date.day)+' to '+
          str(day_16.year)+'/'+str(day_16.month)+'/'+str(day_16.day))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

# Create animatation of the overlap of all methods
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig, ax = plt.subplots()
fig,ax = plt.subplots(1, 1,figsize=(12, 6))
ims=[]

for i in range(len(fnames)):
    MODIS_ndvi_map=rio.open(fnames[i], "r")
    MODIS_ndvi_val=MODIS_ndvi_map.read(1)
    MODIS_ndvi=data_solve_nan_ref(MODIS_ndvi_val)
    date=dfs_names[i][-11:-4]

    image_date= datetime(int(date[0:4]),1,1)+timedelta(days=int(date[4:7]))
    day_16 = image_date + timedelta(days=16)
    fig.patch.set_alpha(0)
    
    t=plt.imshow(MODIS_ndvi, extent=bbox,cmap='RdYlGn',zorder=1)
    #t = plt.plot(range(i, i + 5))
    year=image_date.year
    month=image_date.month
    day=image_date.day
    title = plt.text(0.5,1.01,'16 day composite ndvi from ' +str(image_date.year)+'/'+str(image_date.month)+'/'+str(image_date.day)+' to '+
              str(day_16.year)+'/'+str(day_16.month)+'/'+str(day_16.day), ha="center",va="bottom",
                     transform=ax.transAxes, fontsize=12)
    plt.grid(zorder=0)
    #plt.title('Chl-a')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    ims.append([t,title,])
plt.colorbar(label='ndvi')    
ani = animation.ArtistAnimation(fig, ims, interval=1500, blit=True,
                              repeat_delay=2000)
plt.show()
ani.save('D:/project/ndvi_hawaii/gap_filled ndvi data/ndvi_animation_hawaii.gif', writer='imagemagick')