###############################
# Start of file:
# Author(s): Ellen Dyer (2021)
# Contact: ellen.dyer@ouce.ox.ac.uk
# calculation of wet season onset and cessation
###############################
#    These functions find the onset and cessation of the wet season/s
#    The methodology is based on that of Liebmann et al. (2012) Journal of Climate
#    This methodology closely follows on the methodology described in Dunning et al. (2016)

import numpy as np
import xarray as xr
from scipy import signal
from math import log
import matplotlib.pyplot as plt

def mult_one_season(Cs,C):
    inds_max = signal.argrelextrema(Cs, np.greater)[0]
    inds_min = signal.argrelextrema(Cs, np.less)[0]
    on_1 = inds_min[np.where(Cs[inds_min]==np.min(Cs[inds_min]))]
    ce_1 = inds_max[np.where(Cs[inds_max]==np.max(Cs[inds_max]))]
    ON1,OFF1 = C[on_1].dayofyear.values, C[ce_1].dayofyear.values
    return on_1,ce_1,ON1,OFF1

def simple_one_season(Cs,C):
    try:
      on_1 = signal.argrelextrema(Cs, np.less)[0][0]
      ce_1 = signal.argrelextrema(Cs, np.greater)[0][0]
      ON1,OFF1 = C[on_1].dayofyear.values, C[ce_1].dayofyear.values
    except:
      on_1,ce_1 = np.nan,np.nan
      ON1,OFF1 = np.nan,np.nan
    return on_1,ce_1,ON1,OFF1


def extremes_max(Cs):
    """
    returns largest val l and second largest val sl in list
    """
    inds_max = signal.argrelextrema(Cs, np.greater)[0]
    # don't want to double count the year looped part of ts
    inds_max = inds_max[np.where(inds_max > 85)[0]]
    #print('inds_max',inds_max)
    #print('Cs inds_max',Cs[inds_max])
    
    new_max = []
    for ind in inds_max:
      if Cs[ind-4] < Cs[ind] and Cs[ind+4] < Cs[ind]:
        new_max.append(ind)
    #print('new_max',new_max)

#    l = max(Cs[inds_max])
#    sl = max(item for item in Cs[inds_max] if item < l)
#    #print('Cs maxs',l,sl)
#    maxs = []
#    maxs.append(inds_max[np.where(Cs[inds_max]==l)[0][0]])
#    maxs.append(inds_max[np.where(Cs[inds_max]==sl)[0][0]])
#    print('maxs presort', maxs)
#    maxs.sort()
#    print('maxs postsort', maxs)
#    return maxs
    return new_max

def extremes_min(Cs):
    """
    returns smallest val s and second smallest val ss in list
    """
    inds_min = signal.argrelextrema(Cs, np.less)[0]
    # don't want to double count the year looped part of ts
    maxmin = len(Cs)-50
    inds_min = inds_min[np.where(inds_min < maxmin)[0]]
    #print('inds_min',inds_min)
    #print('Cs inds_min',Cs[inds_min])
    
    new_min = []
    for ind in inds_min:
      if Cs[ind-4] > Cs[ind] and Cs[ind+4] > Cs[ind]:
        new_min.append(ind)
    #print('new_min',new_min)

#    l = min(Cs[inds_min])
#    sl = min(item for item in Cs[inds_min] if item > l)
#    mins = []
#    mins.append(inds_min[np.where(Cs[inds_min]==l)[0][0]])
#    mins.append(inds_min[np.where(Cs[inds_min]==sl)[0][0]])
#    print('mins presort', mins)
#    mins.sort()
#    print('mins postsort', mins)
#    return mins
    return new_min

def find_closest_two_power(length):
    """
    This function finds the greatest power of 2 smaller than length
    It will return the power smaller to reduce zero padding 
    """
    power = log(length,2)
    new_length = 2**int(round(power))
    return new_length


def fourier_analysis(time_series, detrend=False):
    """ This function performs fourier analysis of a time series
    The time_series is a 1D array containing points at every time step
    Length determines the length of the array to be output i.e. are we using zero-padding
    If detrend is set to True then the data will be detrended before fourier analysis is applied
    The value detrend is set to determines the order of the polynomial removed
    dt is the time step between each point. 
    """ 
 
    length = find_closest_two_power(len(time_series))
    dt = 1.0/len(time_series.groupby('time.dayofyear').mean('time'))

    # Detrend the data as necessary
    if detrend != False:
        t = np.arange(len(time_series))
        p = np.polyfit(t,time_series, detrend)
        pv = np.polyval(p,t)
        time_series = time_series - pv
    
    # Divide by the standard deviation
    time_series = time_series/np.std(time_series)

    # FFT procedure
    F = np.fft.fft(time_series, n=length)
    N = length
    w = np.fft.fftfreq(N, dt)
 
    # Focus on positive frequencies
    ind_pos = np.where(w>=0)
    freqs = w[ind_pos]
    mag = np.abs(F[ind_pos])/N
    #phase = np.angle(F[ind_pos])  

    # Return freqs, mag
    return freqs, mag


def power_mag(freq,mag):

    #sensitivity
    sens1,sens2,sens3 = 0.02,0.02,0.02
    #sens1,sens2,sens3 = 0.02,0.015,0.015
    # Find power at 1/2/3 cycles per year
    power_storage1 = round(np.mean(mag[np.where(np.abs(freq-1)<=sens1)[0]]),2)
    power_storage2 = round(np.mean(mag[np.where(np.abs(freq-2)<=sens2)[0]]),2)                   
    power_storage3 = round(np.mean(mag[np.where(np.abs(freq-3)<=sens3)[0]]),2)
    print('p1',power_storage1)
    print('p2',power_storage2)
    print('p3',power_storage3)

    # Calculate the ratio of 2 to 1
    power_storage_ratio = power_storage2/power_storage1
    power_ratio3 = power_storage3/power_storage1
    #print('power ratio', power_storage_ratio)
    #print('power ratio 3', power_ratio3)
    
    return power_storage_ratio, power_ratio3, power_storage1, power_storage2


def onset_cess_base(tseries,extra_season,rolldays=None):
     """
     """
     ts_day = tseries.groupby('time.dayofyear').mean('time')
     max_day = ts_day.dayofyear.values[-1]     
     Qb = ts_day.mean('dayofyear')
     clim_anom = ts_day - Qb
     Csum = np.cumsum(clim_anom)
  
     CsumA = Csum.sel(dayofyear=slice(max_day-50,max_day))
     CsumB = Csum.sel(dayofyear=slice(1,85))
     C = xr.concat([CsumA,Csum,CsumB], dim="dayofyear")
  
     wlen = 30
     N = 2.0  # order of butterworth filter
     W = 1.0/wlen  # cutoff frequency
     b, a = signal.butter(N,W,analog=False)
     Cs = signal.filtfilt(b,a,C,method='gust')
     
#     plt.plot(np.arange(0,len(Cs)),Cs)
#     plt.plot(np.arange(0,len(C)),C)
#     plt.show()
#     plt.clf()
  
     # onset season 1
     if extra_season == 'no':
       if (len(signal.argrelextrema(Cs, np.less)[0]) > 1 and len(signal.argrelextrema(Cs, np.greater)[0]) > 1):
         on_1,ce_1,ON1,OFF1 = mult_one_season(Cs,C)
       else:
         on_1,ce_1,ON1,OFF1 = simple_one_season(Cs,C)
         on_2,ce_2,ON2,OFF2 = np.nan,np.nan,np.nan,np.nan

     # onset season 2
     if extra_season=='two' or extra_season=='three':
       # check for false positives
       # if there aren't multiple relative extrema:
       if (len(signal.argrelextrema(Cs, np.less)[0]) == 1 or \
          len(signal.argrelextrema(Cs, np.greater)[0]) == 1):
         on_1,ce_1,ON1,OFF1 = simple_one_season(Cs,C)
         on_2,ce_2,ON2,OFF2 = np.nan,np.nan,np.nan,np.nan
         on_3,ce_3,ON3,OFF3 = np.nan,np.nan,np.nan,np.nan
       # if there are multiple relative extrema
       else:
         on_list = extremes_min(Cs)
         ce_list = extremes_max(Cs)
         #print('on list',on_list)
         #print('ce list',ce_list)

         # if all the mins are greater than maxs or vice versa
         # there is really only one rainy season
         if (extra_season=='two') or (extra_season=='three'):
           if (min(ce_list) > max(on_list)) or (min(on_list) > max(ce_list)):
             on_1,ce_1,ON1,OFF1 = mult_one_season(Cs,C)
             on_2,ce_2,ON2,OFF2 = np.nan,np.nan,np.nan,np.nan
             on_3,ce_3,ON3,OFF3 = np.nan,np.nan,np.nan,np.nan
           # otherwise continue on and find first relative minimum
           else:
             on_1 = on_list[0]
             # find the first max after the onset (don't worry about looping into 
             # around year for this season
             ce_1 = ce_list[np.where(ce_list>on_1)[0][0]]
             ON1,OFF1 = C[on_1].dayofyear.values, C[ce_1].dayofyear.values
             # if there is another min after season 1 cessation, choose the first one
             if len(np.where(on_list>ce_1)[0]) >= 1:
               on_2 = on_list[np.where(on_list>ce_1)[0][0]]
               # if there is a second season onset and the first max
               # is after second season onset select this
               if np.isnan(on_2) == False:
                 if ce_list[1] > on_2:
                   ce_2 = ce_list[1]
                   ON2,OFF2 = C[on_2].dayofyear.values, C[ce_2].dayofyear.values
                   on_3,ce_3 = np.nan,np.nan
                   ON3,OFF3 = np.nan,np.nan
                 # if neither of these options work there is an issue, set season
                 # onset and cessation to nan
                 else:
                   on_2,ce_2 = np.nan,np.nan
                   on_3,ce_3 = np.nan,np.nan
                   ON2,OFF2 = np.nan,np.nan
                   ON3,OFF3 = np.nan,np.nan
         if extra_season=='three':
           if (min(ce_list) > max(on_list)) or (min(on_list) > max(ce_list)):
             on_1,ce_1,ON1,OFF1 = mult_one_season(Cs,C)
             on_2,ce_2,ON2,OFF2 = np.nan,np.nan,np.nan,np.nan
           # otherwise continue on and find first relative minimum
           else:
             on_1 = on_list[0]
             # find the first max after the onset (don't worry about looping into 
             # around year for this season
             ce_1 = ce_list[np.where(ce_list>on_1)[0][0]]
             ON1,OFF1 = C[on_1].dayofyear.values, C[ce_1].dayofyear.values
             # if there is another min after season 1 cessation, choose the first one
             if len(np.where(on_list>ce_1)[0]) >= 1:
               on_2 = on_list[np.where(on_list>ce_1)[0][0]]
               # if there is a second season onset and the first max
               # is after second season onset select this
               if np.isnan(on_2) == False:
                 if ce_list[1] > on_2:
                   ce_2 = ce_list[1]
                   ON2,OFF2 = C[on_2].dayofyear.values, C[ce_2].dayofyear.values
                 else:
                   on_2,ce_2,on_3,ce_3 = np.nan,np.nan,np.nan,np.nan
                   ON2,OFF2,ON3,OFF3 = np.nan,np.nan,np.nan,np.nan
             if len(np.where(on_list>ce_2)[0]) >= 1:
               on_3 = on_list[np.where(on_list>ce_2)[0][0]]
               if np.isnan(on_3) == False:
                 if ce_list[2] > on_3:
                   ce_3 = ce_list[2]
                   ON3,OFF3 = C[on_3].dayofyear.values, C[ce_3].dayofyear.values
                 # if neither of these options work there is an issue, set season
                 # onset and cessation to nan
                 else:
                   on_3,ce_3 = np.nan,np.nan
                   ON3,OFF3 = np.nan,np.nan
             else:
               on_3,ce_3 = np.nan,np.nan
               ON3,OFF3 = np.nan,np.nan

       if np.isnan([on_1,ce_1]).all() == False:
         plt.plot(np.arange(0,len(Cs)),Cs)
         plt.xticks(np.arange(0,len(Cs))[::30],\
                  C.dayofyear.values[::30],rotation=70)
         plt.scatter(on_1,Cs[on_1],s=50,alpha=0.5,c='orange')
         plt.scatter(ce_1,Cs[ce_1],s=50,alpha=0.5,c='green')
         plt.axvline(on_1,ls='--',c='black')
         plt.axvline(ce_1,ls='--',c='black')
         if np.isnan([on_2,ce_2]).all() == False:
           plt.scatter(on_2,Cs[on_2],s=50,alpha=0.5,c='orange')
           plt.scatter(ce_2,Cs[ce_2],s=50,alpha=0.5,c='green')
           plt.axvline(on_2,ls='--',c='black')
           plt.axvline(ce_2,ls='--',c='black')
         if np.isnan([on_3,ce_3]).all() == False:
           plt.scatter(on_3,Cs[on_3],s=50,alpha=0.5,c='orange')
           plt.scatter(ce_3,Cs[ce_3],s=50,alpha=0.5,c='green')
           plt.axvline(on_3,ls='--',c='black')
           plt.axvline(ce_3,ls='--',c='black')
         plt.xlabel('day of year')
         plt.ylabel('Cummulative daily anomaly (mm)')
         #plt.savefig('plots/base_water_year.png',bbox_inches='tight',dpi=200)
         plt.show()
         plt.clf()

       return ON1,OFF1,ON2,OFF2,ON3,OFF3

     else:

       if np.isnan([on_1,ce_1]).all() == False:
         plt.plot(np.arange(0,len(Cs)),Cs)
         plt.xticks(np.arange(0,len(Cs))[::30],\
                  C.dayofyear.values[::30],rotation=70)
         plt.scatter(on_1,Cs[on_1],s=50,alpha=0.5,c='orange')
         plt.scatter(ce_1,Cs[ce_1],s=50,alpha=0.5,c='green')
         plt.axvline(on_1,ls='--',c='black')
         plt.axvline(ce_1,ls='--',c='black')
         plt.xlabel('day of year')
         plt.ylabel('Cummulative daily anomaly (mm)')
         #plt.savefig('plots/base_water_year.png',bbox_inches='tight',dpi=200)
         plt.show()
         plt.clf()

       return ON1,OFF1
  

def onset_cess_year(tseries,on_b,ce_b,window):
     #print(on_b,ce_b)
     ON = []
     OFF = []
     YEARS= tseries.groupby('time.year').mean('time')['year'].values
     Y1,Y2 = YEARS[0],YEARS[-1]
     ts_day = tseries.groupby('time.dayofyear').mean('time')
     max_day = ts_day.dayofyear.values[-1]
     Qb = ts_day.mean('dayofyear')
     for Y in range(int(Y1+1),Y2):
       if ce_b > on_b:
         Qi = tseries.sel(time=slice(str(Y-1)+'-10-30',str(Y+1)+'-02-28'))
         doy1 = Qi['time.dayofyear'][0].values
         Qi = Qi.isel(time=slice(int(max_day-doy1+on_b-window),int(max_day-doy1+ce_b+window)))
       else:
         Qi = tseries.sel(time=slice(str(Y)+'-01-01',str(Y+1)+'-12-30'))
         Qi = Qi.isel(time=slice(int(on_b-window),int(max_day+ce_b+window)))
       Cd = Qi - Qb
       C = np.cumsum(Cd)
     
       wlen = 10
       N = 2.0  # order of butterworth filter
       W = 1.0/wlen  # cutoff frequency
       b, a = signal.butter(N,W,analog=False)
       Cs = signal.filtfilt(b,a,C,method='gust')
  
       # dig into the nature of the maxs and mins if they are right at the beginning
       # of the window - then pick the relative max and mins using the slightly smoothed
       # timeseries and picking the absolute max and min of the relative maxs and mins 
       # I think that doing a smoothing of 10 days is the same thing as figuring out if the 
       # 4 days around the max or min are smaller or larger respectively.
       if (np.where(C==np.max(C))[0] < 15) or (np.where(C==np.min(C))[0] > len(C)-15):
         inds_max = signal.argrelextrema(Cs, np.greater)[0]
         inds_min = signal.argrelextrema(Cs, np.less)[0]
         #cessation
         if len(inds_max) > 1:
           ind_off = inds_max[np.where(Cs[inds_max]==np.max(Cs[inds_max]))]         
         elif len(inds_max) == 1: 
           ind_off = signal.argrelextrema(Cs, np.greater)[0][0]         
         else:
           ind_off = 0
         #onset
         if len(inds_min) > 1:
           ind_on = inds_min[np.where(Cs[inds_min]==np.min(Cs[inds_min]))]         
         elif len(inds_min) == 1:
           ind_on = signal.argrelextrema(Cs, np.less)[0][0]         
         else:
           ind_on = 0
         if ind_off > ind_on:        
           on=Cd['time.dayofyear'].isel(time=ind_on).values
           off=Cd['time.dayofyear'].isel(time=ind_off).values
           ON.append(on.item())
           OFF.append(off.item())
         else:
           #print(Y,'failed')
           ON.append(np.nan)
           OFF.append(np.nan)
       # if there isn't an issue with the max and min points being the end points
       # then figure out if the min is after tha max - if so, the season has failed
       elif (np.where(C==np.min(C))[0]) > (np.where(C==np.max(C))[0]):
         ON.append(np.nan)
         OFF.append(np.nan)
         #print(Y,'failed')
       # else, everything is great and the normal method can be applied
       else:
         ind_on = np.where(C==np.min(C))[0][0]
         ind_off = np.where(C==np.max(C))[0][0]
         on=Cd['time.dayofyear'].isel(time=ind_on).values
         off=Cd['time.dayofyear'].isel(time=ind_off).values
         ON.append(on.item())
         OFF.append(off.item())

#       plt.title(Y)
#       plt.plot(np.arange(1,len(C)+1),C)
#       plt.plot(np.arange(1,len(Cs)+1),Cs)
#       plt.xticks(np.arange(1,len(C)+1)[::10],\
#              Cd['time.dayofyear'].values[::10],rotation=70)
#       plt.scatter(ind_on,C[ind_on],s=50,alpha=0.5,c='orange')
#       plt.scatter(ind_off,C[ind_off],s=50,alpha=0.5,c='green')
#       plt.axvline(ind_on,ls='--',c='black')
#       plt.axvline(ind_off,ls='--',c='black')
#       plt.show()
#       plt.clf()

     return ON,OFF

def xarray_on_cess(xarr):
     """
     """
     on1 = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     ce1 = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     on2 = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     ce2 = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     on3 = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     ce3 = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     prat = np.full((np.shape(xarr.values[0,:,:])),np.nan)
     on1_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1,:,:])),np.nan)
     ce1_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1,:,:])),np.nan)
     on2_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1,:,:])),np.nan)
     ce2_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1,:,:])),np.nan)
     on3_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1,:,:])),np.nan)
     ce3_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1,:,:])),np.nan)
     cla = 0
     for la in xarr.lat.values:
       clo = 0
       for lo in xarr.lon.values:
         #print(la,lo)
         if np.isnan(xarr.sel(lat=la,lon=lo)).all():
           pass
         else:
           freq, mag = fourier_analysis(xarr.sel(lat=la,lon=lo))
           power_ratio, power_ratio3, power_1, power_2 = power_mag(freq,mag)
           prat[cla,clo] = power_ratio
           #if power_ratio >= 1.0:
           if power_ratio >= 0.8:
             #print('there is a second season')
             if power_ratio3 >= power_ratio:
               #print('there is a third season')
                   #print('there is a second season')
               on1[cla,clo],ce1[cla,clo],on2[cla,clo],ce2[cla,clo],on3[cla,clo],ce3[cla,clo] = onset_cess_base(xarr.sel(lat=la,lon=lo),extra_season='three')
               if np.isnan([on3[cla,clo],ce3[cla,clo]]).all() == False:
                 on3_y[:,cla,clo],ce3_y[:,cla,clo] = onset_cess_year(xarr.sel(lat=la,lon=lo),on3[cla,clo],ce3[cla,clo],window=20)
               if np.isnan([on2[cla,clo],ce2[cla,clo]]).all() == False:
                 on2_y[:,cla,clo],ce2_y[:,cla,clo] = onset_cess_year(xarr.sel(lat=la,lon=lo),on2[cla,clo],ce2[cla,clo],window=20)
               if np.isnan([on1[cla,clo],ce1[cla,clo]]).all() == False:
                 on1_y[:,cla,clo],ce1_y[:,cla,clo] = onset_cess_year(xarr.sel(lat=la,lon=lo),on1[cla,clo],ce1[cla,clo],window=20)
             else:
               on1[cla,clo],ce1[cla,clo],on2[cla,clo],ce2[cla,clo],on3[cla,clo],ce3[cla,clo] = onset_cess_base(xarr.sel(lat=la,lon=lo),extra_season='two')
               if np.isnan([on2[cla,clo],ce2[cla,clo]]).all() == False:
                 on2_y[:,cla,clo],ce2_y[:,cla,clo] = onset_cess_year(xarr.sel(lat=la,lon=lo),on2[cla,clo],ce2[cla,clo],window=20)
               if np.isnan([on1[cla,clo],ce1[cla,clo]]).all() == False:
                 on1_y[:,cla,clo],ce1_y[:,cla,clo] = onset_cess_year(xarr.sel(lat=la,lon=lo),on1[cla,clo],ce1[cla,clo],window=20)
           else:
             on1[cla,clo],ce1[cla,clo] = onset_cess_base(xarr.sel(lat=la,lon=lo),extra_season='no')
             if np.isnan([on1[cla,clo],ce1[cla,clo]]).all() == False:
               on1_y[:,cla,clo],ce1_y[:,cla,clo] = onset_cess_year(xarr.sel(lat=la,lon=lo),on1[cla,clo],ce1[cla,clo],window=50)
         clo = clo + 1
       cla = cla + 1
     on1 = xarr.mean('time').copy(data=on1)
     ce1 = xarr.mean('time').copy(data=ce1)
     on2 = xarr.mean('time').copy(data=on2)
     ce2 = xarr.mean('time').copy(data=ce2)
     on3 = xarr.mean('time').copy(data=on3)
     ce3 = xarr.mean('time').copy(data=ce3)
     power_ratio = xarr.mean('time').copy(data=prat)
     on1_years = xarr.groupby('time.year').mean('time')[1:-1,:,:].copy(data=on1_y)
     ce1_years = xarr.groupby('time.year').mean('time')[1:-1,:,:].copy(data=ce1_y)
     on2_years = xarr.groupby('time.year').mean('time')[1:-1,:,:].copy(data=on2_y)
     ce2_years = xarr.groupby('time.year').mean('time')[1:-1,:,:].copy(data=ce2_y)
     on3_years = xarr.groupby('time.year').mean('time')[1:-1,:,:].copy(data=on3_y)
     ce3_years = xarr.groupby('time.year').mean('time')[1:-1,:,:].copy(data=ce3_y)
     #calculate dry season length here
     #return power_ratio,on1_base,ce1_base,on2_base,ce2_base,on3_base,ce3_base,on1_years,ce1_years,on2_years,ce2_years,on3_years,ce3_years
     return power_ratio,on1,ce1,on2,ce2,on3,ce3,on1_years,ce1_years,on2_years,ce2_years,on3_years,ce3_years



def xarray_on_cess_point(xarr):
     """
     """
     on1_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1])),np.nan)
     ce1_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1])),np.nan)
     on2_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1])),np.nan)
     ce2_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1])),np.nan)
     on3_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1])),np.nan)
     ce3_y = np.full((np.shape(xarr.groupby('time.year').mean('time')[1:-1])),np.nan)

     freq, mag = fourier_analysis(xarr)
     power_ratio, power_ratio3,  power_1, power_2 = power_mag(freq,mag)
     prat = power_ratio
     #if power_ratio >= 1.0:
     if power_ratio >= 0.8:
       #print('there is a second season')
       if power_ratio3 >= power_ratio:
         #print('there is a third season')
         on1,ce1,on2,ce2,on3,ce3 = onset_cess_base(xarr,extra_season='three')
         if np.isnan([on2,ce2]).all() == False:
           on2_y[:],ce2_y[:] = onset_cess_year(xarr,on2,ce2,window=20)
         if np.isnan([on3,ce3]).all() == False:
           on3_y[:],ce3_y[:] = onset_cess_year(xarr,on3,ce3,window=20)
         if np.isnan([on1,ce1]).all() == False:
           on1_y[:],ce1_y[:] = onset_cess_year(xarr,on1,ce1,window=20)
       else:
         on1,ce1,on2,ce2,on3,ce3 = onset_cess_base(xarr,extra_season='two')
         if np.isnan([on2,ce2]).all() == False:
           on2_y[:],ce2_y[:] = onset_cess_year(xarr,on2,ce2,window=20)
         if np.isnan([on1,ce1]).all() == False:
           on1_y[:],ce1_y[:] = onset_cess_year(xarr,on1,ce1,window=20)
     else:
       on1,ce1 = onset_cess_base(xarr,extra_season='no')
       on2,ce2 = np.nan,np.nan
       on3,ce3 = np.nan,np.nan
       if np.isnan([on1,ce1]).all() == False:
         on1_y[:],ce1_y[:] = onset_cess_year(xarr,on1,ce1,window=50)

     power_ratio = xarr.mean('time').copy(data=prat)
     on1_years = xarr.groupby('time.year').mean('time')[1:-1].copy(data=on1_y)
     ce1_years = xarr.groupby('time.year').mean('time')[1:-1].copy(data=ce1_y)
     on2_years = xarr.groupby('time.year').mean('time')[1:-1].copy(data=on2_y)
     ce2_years = xarr.groupby('time.year').mean('time')[1:-1].copy(data=ce2_y)
     on3_years = xarr.groupby('time.year').mean('time')[1:-1].copy(data=on3_y)
     ce3_years = xarr.groupby('time.year').mean('time')[1:-1].copy(data=ce3_y)
     #calculate dry season length here
     return power_ratio,on1,ce1,on2,ce2,on3,ce3,on1_years,ce1_years,on2_years,ce2_years,on3_years,ce3_years

