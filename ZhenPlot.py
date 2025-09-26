import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot

import warnings
warnings.filterwarnings("ignore")

from geemap import cartoee as cee
import ee
ee.Authenticate()
ee.Initialize()

plt.rcParams['font.sans-serif'] = ['Arial']

def add_basemap(ax,xticks=True,yticks=True,xrange=np.arange(-180,180,20),yrange=np.arange(-90,90,20),fontsize=14,ocean_color = '#ffffff'):
    """
    Add land, ocean, and national border to the map.
    args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object to add basemap overlay to
        xticks (bool, optional): whether showing longitude ticks
        yticks (bool, optional): whether showing latitude ticks
        xrange (np.array, optional): a series of longitude to showing as ticks.
        yrange (np.array, optional): a series of latitude to showing as ticks.

    raises:
        ValueError: If `ax` is not of type cartopy.mpl.geoaxes.GeoAxesSubplot
    """
    if type(ax) not in [GeoAxes, GeoAxesSubplot]:
        raise ValueError(
            "provided axes not of type cartopy.mpl.geoaxes.GeoAxes "
            "or cartopy.mpl.geoaxes.GeoAxesSubplot"
        )
    
    ax.add_feature(cfeature.LAND,color='#dddddd')
    ax.add_feature(cfeature.OCEAN,color=ocean_color)
    ax.patch.set_facecolor("#f3f4f6")
    ax.add_feature(cfeature.BORDERS,color='#ffffff',lw=0.5,alpha=0.5)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    if yticks:
        ax.set_yticks(yrange, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.yticks(size = fontsize)
    if xticks:
        ax.set_xticks(xrange, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(lon_formatter)
        plt.xticks(size = fontsize)
    ax.set_xlabel('')
    ax.set_ylabel('')

def add_smallmap(fig,axloc,pointloc,markersize = 3):
    ax_inset = fig.add_axes(axloc,projection=ccrs.Orthographic(central_longitude=pointloc[0], central_latitude=pointloc[1]))
    ax_inset.add_feature(cfeature.LAND, zorder=0, edgecolor='none', facecolor='#2e4f4f') 
    ax_inset.set_global()
    ax_inset.plot(pointloc[0], pointloc[1], 'ro', markersize=markersize, transform=ccrs.Geodetic()) 
    ax_inset.gridlines(linestyle='-', color='gray', alpha=0.7, linewidth=0.5)
    return ax_inset

def spine_setting(ax,linewidth=0.7):
    """
    Revising the linewidth of spine
    args:
        ax (matplotlib axes): required matplotlib axes
        linewidth (float): the targeted linewidth
    """
    ax.tick_params(axis='both', which='major', width=linewidth)
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

def plot_landsat(ax,startdate,enddate,loc,
                 vis_params={'bands': ['nir', 'swir1', 'red'],'min': 0.0,'max': 0.3},
                 interval=0.2,zorder=2):
    """
    Add a Landsat image to the map
    args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot | cartopy.mpl.geoaxes.GeoAxes): required cartopy GeoAxesSubplot object to add basemap overlay to
        startdate (str): the start date to get Landsat images
        enddate (str): the end date to get Landsat images
        loc (list): [minx,maxx,miny,maxy].
        vis_params (dict, optional): visualization parameter for plotting the Landsat median composite.
        interval (float,optional): the step of x and y ticks
    """
    def maskL5sr(image):
        qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        saturationMask = image.select('QA_RADSAT').eq(0)
        
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
        thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
        
        return (image.addBands(opticalBands, None, True)
            .addBands(thermalBand, None, True)
            .updateMask(qaMask)
            .updateMask(saturationMask)
            .select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'])
            .rename(['blue','green','red','nir','swir1','swir2'])
            .toFloat())

    def maskL8sr(image):
        qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        saturation_mask = image.select('QA_RADSAT').eq(0)
        
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

        return (image.addBands(optical_bands, None, True)
                .addBands(thermal_bands, None, True)
                .updateMask(qa_mask)
                .updateMask(saturation_mask)
                .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                .rename(['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
                .toFloat())
    minx,maxx,miny,maxy = loc
    extent =  ee.Geometry.Polygon([[[minx,maxy],[minx,miny],[maxx,miny],[maxx,maxy]]])
    
    L4 = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2").filterBounds(extent).map(maskL5sr)
    L5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(extent).map(maskL5sr)
    L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(extent).map(maskL5sr)
    L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(extent).map(maskL8sr)
    L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(extent).map(maskL8sr)
    Landsat = L4.merge(L5).merge(L7).merge(L8).merge(L9)
    bands = ['blue','green','red','nir','swir1','swir2']
    img = Landsat.select(bands).filterDate(startdate,enddate).median().clip(extent)
    cee.add_layer(ax,img,region = [maxx,miny,minx,maxy],vis_params = vis_params,zorder=zorder)
    cee.add_gridlines(ax, interval=interval,linewidth=0)

def generating_extent(midpoint,square=True,xresolution=0.1,yresolution=0.1):
    """
    Generate an extent according to a given point
    args:
        midpoint (list): [lon,lat]
        square (bool, optional): generate a square extent or not
        xresolution (float, optional): the longitudinal resolution
        yresolution (float, optional): the latitudinal resolution
    Returns:
        extent (list): [miny,maxy,minx,maxx]
    """
    y,x = midpoint
    if square:
        if xresolution != yresolution:
            raise ValueError(
                "provided resolutions are not matched for generating a square region"
            )
        else:
            step = xresolution/2
            extent = [y-step,y+step,x-step,x+step]
    else:
        xstep,ystep = xresolution/2,yresolution/2
        extent = [y-ystep,y+ystep,x-xstep,x+xstep]
    return extent

def BivariatePloting(var1,var2,gdf,ax,colors,colorrange=np.arange(0.1,1.0,0.1),cb=False,cax = None):
    """
    Generate a bivariate map
    args:
        var1 (str): name of variable 1,
        var2 (str): name of variable 2,
        gdf (geodataframe): vector file for showing,
        ax (geoaxes): cartopy geoaxes,
        colors (list): [lower left,lower right, upper left, upper right],
        colorrange (np.array): array for reclassify variables,
        cb (bool, optional): whether to show colorbar,
        cax (matplotlib axes, optional): the axes to show colorbar
    Returns:
        ax (geoaxes): cartopy geoaxes
    """
    import matplotlib.colors as mcolors
    from pysal.viz import mapclassify as mc
    gdf['variable1'] = gdf[var1]
    gdf['variable2'] = gdf[var2]
    n_bins = len(colorrange)+1
    classifier1 = mc.UserDefined(gdf['variable1'],  colorrange)
    classifier2 = mc.UserDefined(gdf['variable2'],  colorrange) 
    
    gdf['bin1'] = classifier1.yb  # bin of variable1
    gdf['bin2'] = classifier2.yb  # bin of variable2
    color_array = np.zeros((n_bins, n_bins, 4))
    for i in range(n_bins):
        for j in range(n_bins):
            c00 = mcolors.to_rgba(colors[0][0])  
            c01 = mcolors.to_rgba(colors[0][1])  
            c10 = mcolors.to_rgba(colors[1][0])  
            c11 = mcolors.to_rgba(colors[1][1])  
            c0 = [c00[k] + (c01[k] - c00[k]) * (j / (n_bins-1)) for k in range(4)]
            c1 = [c10[k] + (c11[k] - c10[k]) * (j / (n_bins-1)) for k in range(4)]
            color_array[i, j] = [c0[k] + (c1[k] - c0[k]) * (i / (n_bins-1)) for k in range(4)]

    colors_mapped = [color_array[int(bin2), int(bin1)] for bin1, bin2 in zip(gdf['bin1'], gdf['bin2'])]
    for idx, geom in gdf.iterrows():
        try:
            ax.add_geometries([geom.geometry], crs=ccrs.PlateCarree(), facecolor=colors_mapped[idx], edgecolor='none', zorder=1,alpha=1)
        except:
            print(f"Error plotting geometry at index {idx}")
            continue

    lon_min, lon_max = -180, 180  # 经度范围
    lat_min, lat_max = -41, 31    # 纬度范围
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    if cb:
        cax.imshow(color_array, origin='lower')
        cax.set_xticks([0, n_bins-1])
        cax.set_yticks([0, n_bins-1])
        cax.set_xticklabels(['0', '1%'])
        cax.set_yticklabels(['0', '1%'])
        cax.set_title('Mean annual rate', fontsize=12)
        cax.set_xlabel(var1,c="#5d5a5a")  # 替换为你的变量名
        cax.set_ylabel(var2,c="#5d5a5a")  # 替换为你的变量
    return ax