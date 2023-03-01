###############API CODES TO DOWNLOAD NDVI 
#Set Up the Working Environment
# Import packages 
import requests as r
import getpass, pprint, time, os, cgi, json
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
#print(rio.__version__)
import os, shutil
#plt.rcParams["font.family"] = "Times New Roman"
import time
startTime = time.time()

import sys
 
if len(sys.argv) != 4:
    raise ValueError('Please provide username and password for https://appeears.earthdatacloud.nasa.gov/api and the output path' )
 
username = sys.argv[1]
password = sys.argv[2]


# Set input directory, change working directory
inDir =  sys.argv[3] #'D:/kauai/'           # IMPORTANT: Update to reflect directory on your OS


os.chdir(inDir)                                      # Change to working directory
api = 'https://appeears.earthdatacloud.nasa.gov/api/'  # Set the AρρEEARS API to a variable

# Login
import requests
#response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=('username', 'password'))
token_response= r.post('{}login'.format(api), auth=(username, password)).json() # Insert API URL, call login service, provide credentials & return json

#token_response = response.json()
print(token_response)

# Search and Explore Available Products 
product_response = requests.get('{}product'.format(api)).json()                         # request all products in the product service
print('AρρEEARS currently supports {} products.'.format(len(product_response))) 

products = {p['ProductAndVersion']: p for p in product_response} # Create a dictionary indexed by product name & version
products['MOD13Q1.006']   

prodNames = {p['ProductAndVersion'] for p in product_response} # Make list of all products (including version)
for p in prodNames:                                            # Make for loop to search list of products 'Description' for a keyword                
    if 'Vegetation Indices' in products[p]['Description']:
        pprint.pprint(products[p])   
        
    
prods = ['MOD13Q1.006']     # Start a list for products to be requested, beginning with MCD15A3H.006
prods 

#lst_response  = r.get('https://appeears.earthdatacloud.nasa.gov/api/product/{}'.format(prods[0])).json()  # Request layers for the 2nd product (index 1) in the list: MOD11A2.006

lst_response = r.get('{}product/{}'.format(api, prods[0])).json()
list(lst_response.keys())

#Use the dictionary key '_250m_16_days_NDVI' to see the information for that layer in the response.
lst_response['_250m_16_days_NDVI'] # Print layer response

lst_response['_250m_16_days_NDVI']['Description']

layers = [(prods[0],'_250m_16_days_NDVI')]  # Create tupled list linking desired product with desired l
layers


prodLayer = []
for l in layers:
    prodLayer.append({
            "layer": l[1],
            "product": l[0]
          })
prodLayer


#Submit an Area Request
token = token_response['token']                      # Save login token to a variable
head = {'Authorization': 'Bearer {}'.format(token)}  # Create a header to store token information, needed to submit a request

nps = gpd.read_file('D:/project/ndvi_hawaii/hawaii map shp/kauai_shp/kauai.shp') # Read in shapefile as dataframe using geopandas

print(nps.head())  
nps_gc = nps.to_json()
#nps_gc = nps[nps['isle']=='Lanai'].to_json() # Extract Grand Canyon NP and set to variable
nps_gc = json.loads(nps_gc)                                            # Convert to json format
nps_gc

projections = r.get('{}spatial/proj'.format(api)).json()  # Call to spatial API, return projs as json
projections 

projs = {}                                  # Create an empty dictionary
for p in projections: projs[p['Name']] = p  # Fill dictionary with `Name` as keys
list(projs.keys()) 
projs['geographic']

task_name = 'NDVI'
#input('Enter a Task Name: ') # User-defined name of the task: 'NPS Vegetation Area' used in example
task_type = ['point','area']        # Type of task, area or point
proj = projs['geographic']['Name']  # Set output projection 
outFormat = ['geotiff', 'netcdf4']  # Set output file format type
import pytz
from datetime import datetime, timedelta
hst = pytz.timezone('HST')
today = datetime.today().astimezone(hst)
prev_day = today - timedelta(days=1)
from dateutil.relativedelta import relativedelta
# current date and time
curDT = datetime.now()

# current day
day = curDT.strftime("%d")
print("day:", day)
today.strftime("%d")

# current month
month = curDT.strftime("%m")
print("month:", month)

# current year
year = curDT.strftime("%Y")
print("year:", year)

# current time
time_now = curDT.strftime("%H:%M:%S")
print("time:", time_now)

# current date and time
date_time = curDT.strftime("%m/%d/%Y, %H:%M:%S")
print("date and time:", date_time)


startDate = '07-01-2022'            # Start of the date range for which to extract data: MM-DD-YYYY
endDate =  curDT.strftime("%m/%d/%Y") 
#endDate = '12-31-2021'              # End of the date range for which to extract data: MM-DD-YYYY
recurring = False                   # Specify True for a recurring date range
#yearRange = [2000,2016]            # if recurring = True, set yearRange, change start/end date to MM-DD
task = {'task_type': task_type[1],
    'task_name': task_name,
    'params': {
         'dates': [
         {
             'startDate': startDate,
             'endDate': endDate
         }],
         'layers': prodLayer,
         'output': {
                 'format': {
                         'type': outFormat[0]}, 
                         'projection': proj},
         'geo': nps_gc,
    }}
#Submit a Task Request
task_response = r.post('{}task'.format(api), json=task, headers=head).json()  # Post json to the API task service, return response as json
task_response  

#Retrieve Task Status
params = {'limit': 2, 'pretty': True} # Limit API response to 2 most recent entries, return as pretty json
tasks_response = r.get('{}task'.format(api), params=params, headers=head).json() # Query task service, setting params and header 
tasks_response      

task_id = task_response['task_id']                                               # Set task id from request submission
status_response = r.get('{}status/{}'.format(api, task_id), headers=head).json() # Call status service with specific task ID & user credentials
status_response                                                                 # Print tasks response
   
#call the task service for your request every 20 seconds to check the status of your request.
# Ping API until request is complete, then continue to Section 4
import time
starttime = time.time()
while r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'] != 'done':
    print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])
    time.sleep(20.0 - ((time.time() - starttime) % 20.0))
print(r.get('{}task/{}'.format(api, task_id), headers=head).json()['status'])


#Download a Request [Bundle API]
#destDir = os.path.join(inDir, task_name)                # Set up output directory using input directory and task name
#if not os.path.exists(destDir):os.makedirs(destDir)     # Create the output directory

#Explore Files in Request Output [List Files]

bundle = r.get('{}bundle/{}'.format(api,task_id), headers=head).json()  # Call API and return bundle contents for the task_id as json
bundle  


#Download Files in a Request (Automation) [Download File]
files = {}                                                       # Create empty dictionary
for f in bundle['files']: 
    print(f)
    if f['file_name'][-27:-23]=='NDVI' and f['file_name'].endswith('.tif'):
        print(f)
        files[f['file_id']] = f['file_name']   # Fill dictionary with file_id as keys and file_name as values
files   


for f in files:
    print(f)
    dl = r.get('{}bundle/{}/{}'.format(api, task_id, f), headers=head, stream=True, allow_redirects = 'True')                                # Get a stream to the bundle file
    if files[f].endswith('.tif'):
        filename = files[f].split('/')[1]
    else:
        filename = files[f] 
    filepath = os.path.join(inDir, filename)                                                       # Create output file path
    with open(filepath, 'wb') as f:                                                                  # Write file to dest dir
        for data in dl.iter_content(chunk_size=8192): f.write(data) 
print('Downloaded files can be found at: {}'.format(inDir))

#inDir = 'D:/kauai/'            
dfs_names = sorted(os.listdir(inDir))
#plot the 1st ndvi map to check        
MODIS_ndvi_ini = rio.open(dfs_names[0], "r")
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

ref_geotiff = rio.open(r'D:/project/ndvi_hawaii/hawaii map tif/kauai.tif')

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
Matrix_ndvi_values=np.zeros((len(dfs_names) , len(df1)))-9000

def data_solve_nan(data):
    df_dem=pd.DataFrame (data)
    df_dem_replace=df_dem.replace(-3000,np.nan)
    return df_dem_replace.values           
for t in range(len(dfs_names)):
    MODIS_ndvi_map=rio.open(dfs_names[t], "r")
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

    with rio.open('D:/kauai_scaled/kauai_%s.tif'%dfs_names[t][-19:-12], 'w', **ref_geotiff_meta) as dst:
        dst.write( MODIS_re_2d, 1)    
        
base_dir = r'D:/kauai_scaled'

fnames = [os.path.join(base_dir, fname) for fname in sorted(os.listdir(base_dir))]

dfs_names = sorted(os.listdir(base_dir))
#dfs_names[0][34:41]
MODIS_chla_ini = rio.open(fnames[0], "r")

MODIS_chla_ini_meta = MODIS_chla_ini.profile
MODIS_chla_ini_pixelSizeX, MODIS_chla_ini_pixelSizeY = MODIS_chla_ini.res

MODIS_chla=MODIS_chla_ini.read(1)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
#plt.imshow(ref_geotiff_data, extent=ref_geotiff.bounds, cmap='cubehelix', zorder=1)
plt.imshow(MODIS_chla, extent=MODIS_chla_ini.bounds, cmap='spring_r', zorder=2)
plt.colorbar(label='legend')
plt.grid(zorder=0)
#plt.title('Chl-a')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()


#make imputation with near real time data
Matrix_ndvi_values=np.zeros((len(dfs_names) , len(df1)))-90000
for t in range(len(dfs_names)):   
    print(t)
    MODIS_ndvi_map=rio.open(fnames[t], "r")
    #MODIS_ndvi_map=rio.open(base_dir  +  "/"+filename[t], "r")
    MODIS_ndvi_val=MODIS_ndvi_map.read(1)
    #MODIS_ndvi_val=data_solve_nan_ndvi(MODIS_ndvi_val)
    Matrix_ndvi_values[t,:] = np.reshape(MODIS_ndvi_val, MODIS_ndvi_val.shape[0]*MODIS_ndvi_val.shape[1])


#data_ref=np.reshape(ref_geotiff_data, ref_geotiff_data.shape[0]*ref_geotiff_data.shape[1])   
# mask boundary values in the chl map with -9999 to recognize the nan values in boundary from nan values in the data
#for t in range(len(fnames)):
for t in range(len(dfs_names)):
    Matrix_ndvi_values[t,:][np.isnan(data_ref)==True]=-99999   # -9999 is used to recognize the boundary pixels from missing values (np.nan) in the images

ndvi_values_pd=pd.DataFrame(Matrix_ndvi_values)
ndvi_values_pd.head()
# Drop the columns where all elements are -9999 to first remove all boundary pixels from the data
ndvi_values_drop=ndvi_values_pd.drop(columns=ndvi_values_pd.columns[(ndvi_values_pd == -99999).any()])

# Check rows with complete nan values
index=ndvi_values_drop[ndvi_values_drop.isnull().all(axis=1)].index
# drop selected index
ndvi_values_drop_final=ndvi_values_drop.drop(index)
ndvi_fill_matrix1=ndvi_values_drop_final.copy()
kauai_cols = pd.read_pickle("D:/project/ndvi_hawaii/entire missing columns/kauai_cols.pkl")
ndvi_fill_matrix_final_1=ndvi_fill_matrix1.drop(kauai_cols,axis=1)



i1,c1 = ndvi_fill_matrix_final_1.shape
split_df1=np.array_split(ndvi_fill_matrix_final_1, c1//100+1, axis=1)
import pickle
models = []
with open("D:/project/ndvi_hawaii/trained models/kauai_models.pckl", "rb") as f:
    while True:
        try:
            models.append(pickle.load(f))
        except EOFError:
            break

imputed_data1=[]

for j in range(len(split_df1)):
    print(j)
    data=split_df1[j]
    X1=pd.DataFrame(models[j].transform(data))
    X1.columns=data.columns
    imputed_data1.append(X1)
data_imputed1=pd.concat(imputed_data1, axis=1, ignore_index=False)  


for t in range(len(data_imputed1)):
    ndvi_values_pd=pd.DataFrame(data_imputed1.loc[t])
    image=pd.concat([pd.concat([ndvi_values_pd.T, kauai_cols], axis=1)], axis=1)
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

    with rio.open('D:/kauai_%s.tif'%dfs_names[t][-11:-4], 'w', **ref_geotiff_meta) as dst:
        dst.write( MODIS_re_2d, 1)     