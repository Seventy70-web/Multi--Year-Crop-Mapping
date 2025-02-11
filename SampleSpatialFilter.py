from osgeo import gdal
import numpy as np
import geopandas
from shapely.geometry import Point
import random
import pandas as pd
shp_path = r'data\Wangkui\Wangkui.shp'
shp_df = geopandas.GeoDataFrame.from_file(shp_path, ).iloc[0]['geometry']
ds=gdal.Open(r'data\Wangkui_Crop_2021_30m.tif')
geotransform=ds.GetGeoTransform()
proj=ds.GetProjection()
width = ds.RasterXSize
height =ds.RasterYSize
Lon_array=gdal.Open(r'data\Lon_30m.tif').ReadAsArray().flatten()
Lat_array=gdal.Open(r'data\Lat_30m.tif').ReadAsArray().flatten()
row_array = np.concatenate([np.arange(height).reshape((-1,1))]*width,axis=1).flatten()
col_array = np.transpose(np.concatenate([np.arange(width).reshape((-1,1))]*height,axis=1)).flatten()
dsLandCover=ds.ReadAsArray()
dsLandCover1=dsLandCover.flatten()
num=10000
res=[]

for i in range(4):
    index=np.where(dsLandCover1==i)[0]
    Select=random.sample(range(index.size),num)
    indexSelect=index[Select]
    valuerow=row_array[indexSelect]
    valuecol=col_array [indexSelect]
    valuelon=Lon_array[indexSelect]
    valuelat=Lat_array [indexSelect]
    re=[]
    for j in range(num):
        row=int(valuerow[j])
        col=int(valuecol[j])
        Lon=valuelon[j]
        Lat=valuelat[j]
        point=Point(Lon,Lat)
        if not point.within(shp_df):continue
        edge=2
        if row<5*edge or row>height-5*edge:continue
        if col<5*edge or col>width-5*edge:continue
        r=1
        if i!=0:
            t1=dsLandCover[row-r:row+r+1,col-r:col+r+1]
            if not np.sum(t1)==t1.size*i:continue
        re.append([Lon,Lat,i])
    re=np.array(re)
    FinalNum=2000
    if i==0:FinalNum=2*FinalNum
    if re.shape[0]>FinalNum:
        Select=random.sample(range(re.shape[0]),FinalNum)
    else:
        Select=random.sample(range(re.shape[0]),re.shape[0])
    re=re[Select,:]
    res.append(re)
res=pd.DataFrame(np.concatenate(res),columns=['Lon','Lat','Class'])
res.to_csv('CropSampleSpatialFilter.csv',index=False)

