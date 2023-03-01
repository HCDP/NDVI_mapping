import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
#print(rio.__version__)
import os, shutil
#plt.rcParams["font.family"] = "Times New Roman"
import time
startTime = time.time()

#ref_geotiff = rio.open('/home/trantkt/hpc-test/Oahu_Clip.tif')
ref_geotiff = rio.open(r'D:/project/ndvi_hawaii/hawaii map tif/big.tif')

ref_geotiff_meta = ref_geotiff.profile
ref_geotiff_pixelSizeX, ref_geotiff_pixelSizeY = ref_geotiff.res


# Read coordinates of the ref_geotiff
x_coordinates=np.zeros((ref_geotiff.shape[0] , ref_geotiff.shape[1]))+np.nan
y_coordinates=np.zeros((ref_geotiff.shape[0] , ref_geotiff.shape[1]))+np.nan
for i in range (ref_geotiff.shape[0]):
    #print(i)
    for j in range (ref_geotiff.shape[1]):
        x_coordinates[i,j]=ref_geotiff.xy(i,j)[0]
        y_coordinates[i,j]=ref_geotiff.xy(i,j)[1]

ref_data=ref_geotiff.read(1)

# firtly convert numpy to dataframe to be able to convert nodata values (here:2147483647) to nan values
def data_solve_nan_ref(data):
    df_dem=pd.DataFrame (data)
    #df_dem_replace=df_dem.replace(32767.0,np.nan)
    df_dem_replace=df_dem.replace(123456,np.nan)
    return df_dem_replace.values

ref_geotiff_data=data_solve_nan_ref(ref_data)

data_ref=np.reshape(ref_geotiff_data, ref_geotiff_data.shape[0]*ref_geotiff_data.shape[1])

x_coordinates_re=np.reshape(x_coordinates, x_coordinates.shape[0]*x_coordinates.shape[1])
y_coordinates_re=np.reshape(y_coordinates, y_coordinates.shape[0]*y_coordinates.shape[1])

index=list(range(len(data_ref)))
df1=pd.DataFrame({'index':index,'ref_data': data_ref})
df1['lon']=x_coordinates_re
df1['lat']=y_coordinates_re
df1_drop=df1.dropna()

destDir=r'D:/project/ndvi_hawaii/big'##path to access the ndvi data
fnames = [os.path.join(destDir, fname) for fname in sorted(os.listdir(destDir))]

dfs_names = sorted(os.listdir(destDir))

filename = [] 
for f in range(len(dfs_names)):       
   
    if dfs_names[f].endswith('.tif') and dfs_names[f][26:30]=='NDVI':
        filename.append(dfs_names[f])
#plot the 1st ndvi map to check        
MODIS_ndvi_ini = rio.open(destDir +  "/"+filename[0] , "r")
MODIS_ndvi_ini_meta = MODIS_ndvi_ini.profile
MODIS_ndvi_ini_pixelSizeX, MODIS_ndvi_ini_pixelSizeY = MODIS_ndvi_ini.res

MODIS_ndvi=MODIS_ndvi_ini.read(1)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
#plt.imshow(ref_geotiff_data, extent=ref_geotiff.bounds, cmap='cubehelix', zorder=1)
plt.imshow(MODIS_ndvi, extent=MODIS_ndvi_ini.bounds, cmap='spring_r', zorder=2)
plt.colorbar(label='legend')
plt.grid(zorder=0)
#plt.title('Chl-a')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

Matrix_ndvi_values=np.zeros((len(filename) , len(df1)))-9000

def data_solve_nan(data):
    df_dem=pd.DataFrame (data)
    df_dem_replace=df_dem.replace(-3000,np.nan)
    return df_dem_replace.values           
##produce the real value of ndvi by multipling with scale factor (0.0001)
for t in range(len(filename)):
    MODIS_ndvi_map=rio.open(destDir +  "/"+filename[t], "r")
    MODIS_ndvi_val=MODIS_ndvi_map.read(1)
    MODIS_ndvi_val=data_solve_nan(MODIS_ndvi_val)
    MODIS_ndvi_val=MODIS_ndvi_val*0.0001
    #print(t)
    Matrix_ndvi_values[t,:] = np.reshape(MODIS_ndvi_val, MODIS_ndvi_val.shape[0]*MODIS_ndvi_val.shape[1])
    ndvi_values_pd=pd.DataFrame(Matrix_ndvi_values)
    ndvi_index=ndvi_values_pd.values[t,:]-(df1['lat']/100000000000)
    ndvi_index_df=pd.DataFrame(ndvi_index)
    ndvi_index_df['ndvi']=ndvi_index_df['lat']

    df_concat=pd.concat([df1,ndvi_index_df], axis=1)
    MODIS_re_2d=np.reshape(df_concat['ndvi'].values,(ref_geotiff_data.shape[0],ref_geotiff_data.shape[1]))
    ref_geotiff_meta['dtype'] = "float64"

    with rio.open('D:/project/ndvi_hawaii/big_scaled/big_%s.tif'%filename[t][-19:-12], 'w', **ref_geotiff_meta) as dst:
        dst.write( MODIS_re_2d, 1)    

base_dir = r'D:/project/ndvi_hawaii/bigu_scaled'

fnames = [os.path.join(base_dir, fname) for fname in sorted(os.listdir(base_dir))]
dfs_names = sorted(os.listdir(base_dir))
MODIS_ndvi_test=rio.open(fnames[0], "r")
MODIS_ndvi_ini_meta = MODIS_ndvi_test.profile
MODIS_ndvi_ini_pixelSizeX, MODIS_ndvi_ini_pixelSizeY = MODIS_ndvi_test.res

MODIS_ndvi=MODIS_ndvi_test.read(1)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
#plt.imshow(ref_geotiff_data, extent=ref_geotiff.bounds, cmap='cubehelix', zorder=1)
plt.imshow(MODIS_ndvi, extent=MODIS_ndvi_test.bounds, cmap='spring_r', zorder=2)
plt.colorbar(label='legend')
plt.grid(zorder=0)
#plt.title('Chl-a')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()



Matrix_ndvi_values=np.zeros((len(fnames) , len(df1)))-90000
           
for t in range(len(fnames)):    
    MODIS_ndvi_map=rio.open(fnames[t], "r")
    MODIS_ndvi_val=MODIS_ndvi_map.read(1)
    #MODIS_ndvi_val=data_solve_nan_ndvi(MODIS_ndvi_val)
    Matrix_ndvi_values[t,:] = np.reshape(MODIS_ndvi_val, MODIS_ndvi_val.shape[0]*MODIS_ndvi_val.shape[1])

# mask boundary values in the chl map with -9999 to recognize the nan values in boundary from nan values in the data
for t in range(len(fnames)):
    Matrix_ndvi_values[t,:][np.isnan(data_ref)==True]=-99999   # -9999 is used to recognize the boundary pixels from missing values (np.nan) in the images

ndvi_values_pd=pd.DataFrame(Matrix_ndvi_values)
ndvi_values_pd.head()
# Drop the columns where all elements are -9999 to first remove all boundary pixels from the data
ndvi_values_drop=ndvi_values_pd.drop(columns=ndvi_values_pd.columns[(ndvi_values_pd == -99999).any()])

# Check rows with complete nan values
index=ndvi_values_drop[ndvi_values_drop.isnull().all(axis=1)].index
# drop selected index
ndvi_values_drop_final=ndvi_values_drop.drop(index)
ndvi_fill_matrix=ndvi_values_drop_final.copy()
cols=ndvi_fill_matrix[ndvi_fill_matrix.columns[ndvi_fill_matrix.isnull().all()]]
cols.to_pickle("D:/project/ndvi_hawaii/entire missing columns/big_cols.pkl")
big_cols = pd.read_pickle("D:/project/ndvi_hawaii/entire missing columns/big_cols.pkl")  
ndvi_fill_matrix_final=ndvi_fill_matrix.drop(cols,axis=1)
ndvi_values_miss_large=ndvi_fill_matrix[ndvi_fill_matrix.columns[ndvi_fill_matrix.isnull().all()]]
i,c = ndvi_fill_matrix_final.shape
split_df=np.array_split(ndvi_fill_matrix_final, c//100+1, axis=1)

from scipy.stats import pearsonr
from sklearn import metrics
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor
import pickle
imputed_data=[]
method=[]
for j in range(len(split_df)):
    print(j)
    data=split_df[j]
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0, loss='ls')
    imp = IterativeImputer(estimator=est,missing_values=np.nan, max_iter=5, verbose=1, imputation_order='roman',random_state=0)
    imp.fit(data)
    X=pd.DataFrame(imp.transform(data))
    #X=pd.DataFrame(imp.fit_transform(data))
    X.columns=data.columns
    imputed_data.append(X)
    method.append(imp)
data_imputed=pd.concat(imputed_data, axis=1, ignore_index=False)


for t in range(len(data_imputed)):
    ndvi_values_pd=pd.DataFrame(data_imputed.loc[t])
    image=pd.concat([pd.concat([ndvi_values_pd.T, ndvi_values_miss_large], axis=1)], axis=1)
    image_sort = image.sort_index(axis=1)
    image_index_1=image_sort.loc[t,:]-(df1_drop['lat']/100000000000)
    image_index_1_df=pd.DataFrame(image_index_1,columns=['ndvi'])
    image_df_concat_1=pd.concat([df1,image_index_1_df], axis=1)
    MODIS_re_2d=np.reshape(image_df_concat_1['ndvi'].values,(ref_geotiff_data.shape[0],ref_geotiff_data.shape[1]))
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.imshow(MODIS_re_2d, extent=ref_geotiff.bounds, cmap='cubehelix', zorder=1)
    plt.colorbar(label='NDVI')
    plt.grid(zorder=0)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    ref_geotiff_meta['dtype'] = "float64"

    with rio.open('D:/project/ndvi_hawaii/gap_filled ndvi data/big/big_%s.tif'%dfs_names[t][-11:-4], 'w', **ref_geotiff_meta) as dst:
        dst.write( MODIS_re_2d, 1)    
             
##store trained model by pickle
with open(r"D:/project/ndvi_hawaii/trained models/big_models.pckl", "wb") as f:
    for model in method:
         pickle.dump(model, f)