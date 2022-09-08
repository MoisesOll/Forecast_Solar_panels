import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime as dt
import pickle

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sktime.transformations.series.boxcox import LogTransformer
from sktime.utils.plotting import plot_correlations

from aux_functions import *


def compute_solar_production():
    # Cargamos los datos de los Buildings y Solars
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
        "data/phase_2_data.tsf")

    # Principio y fin de cada edificio, los unimos en una unica lista
    total_df = []
    var_names = loaded_data['series_name'].values

    for var in var_names:
        print('--', var)

        # Obtenemos la fecha de inicio y la cantidad de datos de cada building y panel
        start_time = loaded_data[loaded_data['series_name'] == var]['start_timestamp'].values[0]
        value_data = loaded_data[loaded_data['series_name'] == var]['series_value'].values[0]

        df = pd.DataFrame(data=np.array(value_data).astype(np.float64),
                          index=pd.date_range(start=start_time, periods=len(value_data),
                                              freq='15min'), columns=[var], )

        print(df.shape)
        print(f"Start: {df.index[0]} ,  End: {df.index[-1]}")

        total_df.append(df)

    aggr_df = pd.concat(total_df, join="outer")

    # Anañadimos el mes noviembre al agregado
    nov_dates = pd.date_range(start='2020-11-01', end='2020-11-30 23:45:00', freq='15min')
    # aggr_df = aggr_df.reindex(aggr_df.index.append(nov_dates))

    # Cargamos los datos climatologicos
    weather_df = pd.read_csv('data/ERA5_Weather_Data_Monash.csv', index_col=0, parse_dates=True)

    # Ponemos los datos en la misma frecuencia
    weather_df = weather_df.resample('15min').interpolate('linear')

    # Unimos todo en un mismo DataFrame
    phase2_df = aggr_df.join(weather_df, how='left')

    phase2_df.index = phase2_df.index + dt.timedelta(hours=10)

    # Renombramos las variables quitando caracteres especiales
    phase2_df.rename(columns={'coordinates (lat,lon)': 'coordinates', 'model (name)': 'model',
                              'model elevation (surface)': 'model_elevation',
                              'utc_offset (hrs)': 'utc_offset',
                              'temperature (degC)': 'temperature',
                              'dewpoint_temperature (degC)': 'dewpoint_temperature',
                              'wind_speed (m/s)': 'wind_speed',
                              'mean_sea_level_pressure (Pa)': 'mean_sea_level_pressure',
                              'relative_humidity ((0-1))': 'relative_humidity_01',
                              'surface_solar_radiation (W/m^2)': 'surface_solar_radiation',
                              'surface_thermal_radiation (W/m^2)': 'surface_thermal_radiation',
                              'total_cloud_cover (0-1)': 'total_cloud_cover'}, inplace=True)

   # utilizamos solo solar1 por su mayor número de datos
    solar1 = phase2_df.loc[:, "Solar1"]
    corrupt_dates = solar1.loc["2020-04-15":"2020-05-20"].index
    solar1.drop(index=corrupt_dates, inplace=True)
    solar1 = phase2_df.reset_index()[["index", "Solar1"]].copy()
    solar1.rename({"index": "date"}, inplace=True, axis=1)
    solar1["weekday"] = solar1["date"].apply(lambda x: x.weekday())
    solar1["week"] = solar1["date"].apply(lambda x: x.week)
    solar1["day"] = solar1["date"].apply(lambda x: x.day)
    solar1["year"] = solar1["date"].apply(lambda x: x.year)
    solar1["month"] = solar1["date"].apply(lambda x: x.month)
    seasons = {
        1: "summer",
        2: "fall",
        3: "fall",
        4: "fall",
        5: "winter",
        6: "winter",
        7: "winter",
        8: "spring",
        9: "spring",
        10: "spring",
        11: "summer",
        12: "summer",
    }
    solar1["season"] = solar1["month"].apply(lambda x: seasons[x])
    my_day = dt.date(2018, 12, 31)
    solar1["time"] = solar1["date"].apply(lambda x: datetime.combine(my_day, x.time()))
    start_time = solar1.loc[(solar1["Solar1"].isna() == False)].iat[0, 0]
    solar1 = solar1.loc[solar1.date >= start_time].copy()
    solar1.dropna(inplace=True)

    # Delimitamos los datos para que empiezen desde el primer registro de los eficios
    phase2_df = phase2_df['2019-07-03 04:45:00':'2020-11-30 23:45:00']
    phase2_df = phase2_df[['Building0', 'Building1', 'Building3', 'Building4', 'Building5',
                           'Building6', 'Solar0', 'Solar1', 'Solar2', 'Solar3', 'Solar4', 'Solar5',
                           'temperature', 'dewpoint_temperature',
                           'wind_speed', 'mean_sea_level_pressure',
                           'relative_humidity_01', 'surface_solar_radiation',
                           'surface_thermal_radiation', 'total_cloud_cover']]

    structures = ['Building0', 'Building1', 'Building3', 'Building4', 'Building5',
                  'Building6', 'Solar0', 'Solar1', 'Solar2', 'Solar3', 'Solar4', 'Solar5']
    df_structures = phase2_df[structures]

    weather = ['temperature', 'dewpoint_temperature', 'wind_speed',
               'mean_sea_level_pressure', 'relative_humidity_01',
               'surface_solar_radiation', 'surface_thermal_radiation',
               'total_cloud_cover']
    df_weather = phase2_df[weather]

    numeric_columns = ['Building0', 'Building1', 'Building3', 'Building4', 'Building5',
                       'Building6', 'Solar0', 'Solar1', 'Solar2', 'Solar3', 'Solar4', 'Solar5',
                       'temperature', 'dewpoint_temperature', 'wind_speed',
                       'mean_sea_level_pressure', 'relative_humidity_01',
                       'surface_solar_radiation', 'surface_thermal_radiation',
                       'total_cloud_cover']
    phase2_df[numeric_columns] = phase2_df[numeric_columns].apply(pd.to_numeric)
    df_numerics = phase2_df[numeric_columns]
    with open("/home/jorge/PycharmProjects/solar_power_forecast/data/solar_production.P", "wb") as f:
        pickle.dump(solar1, f)

    with open("/home/jorge/PycharmProjects/solar_power_forecast/data/df_numerics.P", "wb") as f:
        pickle.dump(df_numerics, f)

    with open("/home/jorge/PycharmProjects/solar_power_forecast/data/phase2_df.P", "wb") as f:
        pickle.dump(phase2_df, f)


if __name__ == '__main__':
    compute_solar_production()
