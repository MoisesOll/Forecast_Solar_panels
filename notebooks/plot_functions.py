import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from sktime.utils.plotting import plot_correlations


def plot_daily(solar_production, ax):
    dates = solar_production['date'].between(
        dt.datetime(year=2019, month=10, day=4),
        dt.datetime(year=2020, month=2, day=25),
    )

    dates = (dates |
             solar_production['date'].between(
                 dt.datetime(year=2020, month=5, day=22),
                 dt.datetime(year=2022, month=2, day=28),
             )
             )

    sns.scatterplot(
        x=solar_production.loc[dates, 'time'],
        y=solar_production.loc[dates, "Solar1"],
        alpha=0.2,

    )
    seasons = solar_production["season"].unique()
    colors = {
        "winter": "steelblue",
        "spring": "teal",
        "summer": "goldenrod",
        "fall": "mediumpurple"
    }
    average = {}
    for season in seasons:
        index = solar_production.loc[dates].loc[solar_production["season"] == season].index
        average[season] = solar_production.loc[index].groupby(by="time").mean()
        average[season]["moving_avg"] = average[season]["Solar1"].rolling(5).mean()
        ax.plot(
            average[season].index,
            average[season]["moving_avg"],
            alpha=1,
            label=season,
            color=colors[season],
            linewidth=3
        )

    ax.legend()
    myFmt = mdates.DateFormatter('%H')
    ax.xaxis.set_major_formatter(myFmt)

    return ax


def plot_daily_building3(df, ax):
    sns.set()
    solar_production = df.reset_index()[["Building3", "index"]]
    solar_production.dropna(inplace=True)
    my_day = dt.date(2018, 12, 31)
    solar_production.reset_index(inplace=True)
    solar_production.rename({"index": "date"}, inplace=True, axis=1)
    solar_production["weekday"] = solar_production["date"].apply(lambda x: x.weekday())
    solar_production["week"] = solar_production["date"].apply(lambda x: x.week)
    solar_production["day"] = solar_production["date"].apply(lambda x: x.day)
    solar_production["year"] = solar_production["date"].apply(lambda x: x.year)
    solar_production["month"] = solar_production["date"].apply(lambda x: x.month)
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
    solar_production["season"] = solar_production["month"].apply(lambda x: seasons[x])

    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))
    solar_production.set_index('date')
    ax.scatter(
        x=solar_production.loc[:, 'time'],
        y=solar_production.loc[:, "Building3"],
        s=2,
        alpha=0.5,
    )
    dates = solar_production.index
    seasons = solar_production["season"].unique()
    colors = {
        "winter": "steelblue",
        "spring": "teal",
        "summer": "goldenrod",
        "fall": "mediumpurple"
    }
    average = {}
    # solar_production = solar_production.set_index('date').resample('H').mean()
    for season in seasons:
        index = solar_production.loc[solar_production["season"] == season].index
        average[season] = solar_production.loc[index].groupby(by="time").mean()
        average[season] = average[season].resample('H').mean()
        average[season]["moving_avg"] = average[season]["Building3"].rolling(2).mean()
        ax.plot(
            average[season].index,
            average[season]["moving_avg"],
            alpha=1,
            label=season,
            linewidth=3,
            color=colors[season],
        )

    ax.legend()
    myFmt = mdates.DateFormatter('%H')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Consumo")
    return ax
    # COMMENTS: Data is biased

def plot_month_acf(df):
    sns.set()
    solar_production = df.loc["06-01-2020":"08-01-2020"].reset_index()[["Building3", "index"]]
    solar_production.dropna(inplace=True)
    my_day = dt.date(2018, 12, 31)
    # solar_production.reset_index(inplace=True)
    solar_production.set_index("index", inplace=True)
    solar_production = solar_production.resample(rule="D").mean()

    # solar_production.rename({"index": "date"}, inplace=True, axis=1)
    # solar_production["weekday"] = solar_production["date"].apply(lambda x: x.weekday())
    # solar_production["week"] = solar_production["date"].apply(lambda x: x.week)
    # solar_production["day"] = solar_production["date"].apply(lambda x: x.day)
    # solar_production["year"] = solar_production["date"].apply(lambda x: x.year)
    # solar_production["month"] = solar_production["date"].apply(lambda x: x.month)
    # solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))
    # solar_production.set_index('date')
    g = plot_correlations(solar_production, lags=7)
    g[1][0].tick_params(rotation=-20)
    return g

def plot_month_seasonality(df):
    sns.set()
    solar_production = df.loc["06-01-2020":"08-01-2020"].reset_index()[["Building3", "index"]]
    solar_production.dropna(inplace=True)
    my_day = dt.date(2018, 12, 31)
    # solar_production.reset_index(inplace=True)
    solar_production.set_index("index", inplace=True)
    solar_production = solar_production.resample(rule="D").mean()

    res = seasonal_decompose(solar_production, model="additive", period=7)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    res.trend.plot(ax=ax1, ylabel="trend")
    res.seasonal.plot(ax=ax2, ylabel="seasoanlity")
    res.resid.plot(ax=ax3, ylabel="residual")
    return fig


def plot_weekly_demand(solar_production, ax):
    solar_production = solar_production.set_index('date').resample('3H').mean()
    my_day = dt.date(2018, 12, 31)
    solar_production.reset_index(inplace=True)
    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))

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
    solar_production["season"] = solar_production["month"].apply(lambda x: seasons[x])

    dates = solar_production['date'].between(
        dt.datetime(year=2019, month=10, day=4),
        dt.datetime(year=2020, month=2, day=25),
    )

    dates = (dates |
             solar_production['date'].between(
                 dt.datetime(year=2020, month=5, day=22),
                 dt.datetime(year=2022, month=2, day=28),
             )
             )

    seasons = solar_production["season"].unique()
    colors = {
        "winter": "steelblue",
        "spring": "teal",
        "summer": "goldenrod",
        "fall": "mediumpurple"
    }
    average = {}
    for season in seasons:
        index = solar_production.loc[dates].loc[solar_production["season"] == season].index
        average[season] = solar_production.loc[index].groupby(by=["weekday", "time"]).mean()
        average[season]["moving_avg"] = average[season]["Solar1"].rolling(5).mean()
        average[season].index = [time + dt.timedelta(days=day) for day, time in average[season].index]
        sns.lineplot(
            x=average[season].index,
            y=average[season]["moving_avg"],
            alpha=1,
            label=season,
            color=colors[season],
            ax=ax
        )

    # sns.scatterplot(
    #     x=solar_production.loc[dates, 'weekday'],
    #     y=solar_production.loc[dates, "Solar1"],
    #     alpha=0.2,
    #     size=0.5,
    # )

    ax.legend()
    myFmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_formatter(myFmt)

    return ax
    # COMMENTS: Data is biased because there are some missing months


def plot_weekly_acf(solar_production, figsize=(15, 10)):
    sns.set()
    solar_production = solar_production.set_index('date').resample('H').mean()
    my_day = dt.date(2018, 12, 31)
    solar_production.reset_index(inplace=True)
    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))

    dates = solar_production['date'].between(
        dt.datetime(year=2019, month=10, day=4),
        dt.datetime(year=2020, month=2, day=25),
    )

    dates = (dates |
             solar_production['date'].between(
                 dt.datetime(year=2020, month=5, day=22),
                 dt.datetime(year=2022, month=2, day=28),
             )
             )

    index = solar_production.loc[dates].index
    average = solar_production.loc[index].groupby(by=["weekday", "time"]).mean()
    average["moving_avg"] = average["Solar1"].rolling(5).mean()
    average.index = [time + dt.timedelta(days=day) for day, time in average.index]
    average.dropna(inplace=True)
    g = plot_correlations(average["moving_avg"], lags=24)
    g[1][0].tick_params(rotation=-30)
    return g
    # COMMENTS: Data is biased because there are some missing months


def plot_weekly_seasons_decomp_build3(df, figsize=(15, 10)):
    sns.set()
    solar_production = df.reset_index()[["Building3", "index"]]
    solar_production = solar_production.set_index('index').resample('H').mean()
    my_day = dt.date(2018, 12, 31)
    solar_production.reset_index(inplace=True)
    solar_production.rename({"index": "date"}, inplace=True, axis=1)
    solar_production["weekday"] = solar_production["date"].apply(lambda x: x.weekday())
    solar_production["week"] = solar_production["date"].apply(lambda x: x.week)
    solar_production["day"] = solar_production["date"].apply(lambda x: x.day)
    solar_production["year"] = solar_production["date"].apply(lambda x: x.year)
    solar_production["month"] = solar_production["date"].apply(lambda x: x.month)

    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))
    solar_production.set_index('date')
    dates = solar_production.index
    # dates = solar_production['date'].between(
    #     dt.datetime(year=2019, month=10, day=4),
    #     dt.datetime(year=2020, month=2, day=25),
    # )
    #
    # dates = (dates |
    #          solar_production['date'].between(
    #              dt.datetime(year=2020, month=5, day=22),
    #              dt.datetime(year=2022, month=2, day=28),
    #          )
    #          )

    index = solar_production.loc[dates].index
    average = solar_production.loc[index].groupby(by=["weekday", "time"]).mean()
    average["moving_avg"] = average["Building3"].rolling(5).mean()
    average.index = [time + dt.timedelta(days=day) for day, time in average.index]
    average.dropna(inplace=True)

    res = seasonal_decompose(average["moving_avg"], model="additive", period=24)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    res.trend.plot(ax=ax1, ylabel="trend")
    res.seasonal.plot(ax=ax2, ylabel="seasoanlity")
    res.resid.plot(ax=ax3, ylabel="residual")
    return fig
    # COMMENTS: Data is biased because there are some missing months


def plot_weekly_build3_acf(df):

    sns.set()
    solar_production = df.reset_index()[["Building3", "index"]]
    solar_production = solar_production.set_index('index').resample('H').mean()
    my_day = dt.date(2018, 12, 31)
    solar_production.reset_index(inplace=True)
    solar_production.rename({"index": "date"}, inplace=True, axis=1)
    solar_production["weekday"] = solar_production["date"].apply(lambda x: x.weekday())
    solar_production["week"] = solar_production["date"].apply(lambda x: x.week)
    solar_production["day"] = solar_production["date"].apply(lambda x: x.day)
    solar_production["year"] = solar_production["date"].apply(lambda x: x.year)
    solar_production["month"] = solar_production["date"].apply(lambda x: x.month)

    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))
    solar_production.set_index('date')
    dates = solar_production.index

    index = solar_production.loc[dates].index
    average = solar_production.loc[index].groupby(by=["weekday", "time"]).mean()
    average["moving_avg"] = average["Solar1"].rolling(5).mean()
    average.index = [time + dt.timedelta(days=day) for day, time in average.index]
    average.dropna(inplace=True)
    g = plot_correlations(average["moving_avg"], lags=24)
    g[1][0].tick_params(rotation=-30)
    return g


def plot_weekly_seasons_decomp(solar_production, figsize=(15, 10)):
    sns.set()
    solar_production = solar_production.set_index('date').resample('H').mean()
    my_day = dt.date(2018, 12, 31)
    solar_production.reset_index(inplace=True)
    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))

    dates = solar_production['date'].between(
        dt.datetime(year=2019, month=10, day=4),
        dt.datetime(year=2020, month=2, day=25),
    )

    dates = (dates |
             solar_production['date'].between(
                 dt.datetime(year=2020, month=5, day=22),
                 dt.datetime(year=2022, month=2, day=28),
             )
             )

    index = solar_production.loc[dates].index
    average = solar_production.loc[index].groupby(by=["weekday", "time"]).mean()
    average["moving_avg"] = average["Solar1"].rolling(5).mean()
    average.index = [time + dt.timedelta(days=day) for day, time in average.index]
    average.dropna(inplace=True)

    res = seasonal_decompose(average["moving_avg"], model="additive", period=24)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    res.trend.plot(ax=ax1, ylabel="trend")
    res.seasonal.plot(ax=ax2, ylabel="seasoanlity")
    res.resid.plot(ax=ax3, ylabel="residual")
    return fig
    # COMMENTS: Data is biased because there are some missing months


def plot_yearly(solar_production, ax):
    solar_production = solar_production.set_index('date').resample('D').mean()
    solar_production.reset_index(inplace=True)
    solar_production.rename({"index": "date"}, inplace=True, axis=1)
    solar_production["weekday"] = solar_production["date"].apply(lambda x: x.weekday())
    solar_production["week"] = solar_production["date"].apply(lambda x: x.week)
    solar_production["day"] = solar_production["date"].apply(lambda x: x.day)
    solar_production["year"] = solar_production["date"].apply(lambda x: x.year)
    solar_production["month"] = solar_production["date"].apply
    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))
    
    dates = solar_production['date'].between(
        dt.datetime(year=2019, month=10, day=4),
        dt.datetime(year=2020, month=2, day=25),
    )

    dates = (dates | solar_production['date'].between(
                         dt.datetime(year=2020, month=5, day=22),
                         dt.datetime(year=2022, month=2, day=28),
                     )
             )

    colors = {
        "winter": "steelblue",
        "spring": "teal",
        "summer": "goldenrod",
        "fall": "mediumpurple"
    }
    average = {}

    for season in colors:
        index = solar_production.loc[dates].loc[solar_production["season"] == season].index
        average = solar_production.loc[index].groupby(by=["weekday", "time"]).mean()
        average["moving_avg"] = average["Solar1"].rolling(5).mean()
        average.index = [time + dt.timedelta(days=day) for day, time in average.index]
        sns.lineplot(
            x=average.index,
            y=average["moving_avg"],
            alpha=1,
            label=season,
            color=colors[season],
            ax=ax
        )

    sns.scatterplot(
        x=solar_production.loc[dates, 'weekday'],
        y=solar_production.loc[dates, "Solar1"],
        alpha=0.2,
    )

    ax.legend()
    myFmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_formatter(myFmt)

    return ax
    # COMMENTS: Data is biased because there are some missing months


def plot_histograms(df, columns):
    sns.set()
    # keep total number of subplot
    k = len(df.columns)
    # n = number of chart columns
    n = 5
    m = 4

    # Create figure
    # Create figure
    fig, axes = plt.subplots(
        n, m*2, figsize=(n * 6, m * 3),
        gridspec_kw={'width_ratios': [4, 1] * 4},
    )
    # Iterate through columns, tracking the column name and
    # which number we are at i. Within each iteration, plot
    for i, (name, col) in enumerate(df.iteritems()):
        r, c = i // m, i % m
        ax = axes[r, 2*c]
        col = col.dropna()
        # the histogram
        col.hist(ax=ax, label=name)
        # kde = Kernel Density Estimate plot
        # ax2 = col.plot.kde(ax=ax, secondary_y=False, title=name)
        # ax2.set_ylim(0)
        ax3 = axes[r, 2*c+1]
        ax3.violinplot(col)
        ax.legend()

    # Use tight_layout() as an easy way to sharpen up the layout spacing
    plt.tight_layout()
    return fig, ax


def plot_pairplot(df, figsize=(20, 20)):
    sns.set()
    df_groupped = df.reset_index().groupby(by="index").mean().iloc[:, -13:]
    climate_vs_prod = df_groupped[["temperature", "relative_humidity_01", "surface_solar_radiation", "total_cloud_cover"]]
    climate_vs_prod.loc[:, "Solar"] = df_groupped[["Solar1", "Solar2", "Solar3", "Solar4", "Solar5"]].mean(axis=1)
    g = sns.pairplot(climate_vs_prod[["Solar", "temperature", "relative_humidity_01", "surface_solar_radiation", "total_cloud_cover"]])
    g.fig.set_size_inches(*figsize)
    plt.tight_layout()
    return g


def plot_temp_daily(phase2_df, ax, variable="surface_solar_radiation"):
    solar_production = phase2_df.reset_index()[["index", variable]].copy()
    solar_production.rename({"index": "date"}, inplace=True, axis=1)
    dates = solar_production.index
    solar_production["weekday"] = solar_production["date"].apply(lambda x: x.weekday())
    solar_production["week"] = solar_production["date"].apply(lambda x: x.week)
    solar_production["day"] = solar_production["date"].apply(lambda x: x.day)
    solar_production["year"] = solar_production["date"].apply(lambda x: x.year)
    solar_production["month"] = solar_production["date"].apply(lambda x: x.month)
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
    solar_production["season"] = solar_production["month"].apply(lambda x: seasons[x])
    my_day = dt.date(2018, 12, 31)
    solar_production["time"] = solar_production["date"].apply(lambda x: dt.datetime.combine(my_day, x.time()))
    start_time = solar_production.loc[(solar_production[variable].isna() == False)].iat[0, 0]
    solar_production = solar_production.loc[solar_production.date >= start_time].copy()
    solar_production.dropna(inplace=True)
    sns.scatterplot(
        x=solar_production.loc[dates, 'time'],
        y=solar_production.loc[dates, variable],
        alpha=0.2,
        size=0.5,
    )
    seasons = solar_production["season"].unique()
    colors = {
        "winter": "steelblue",
        "spring": "teal",
        "summer": "goldenrod",
        "fall": "mediumpurple"
    }
    average = {}
    for season in seasons:
        index = solar_production.loc[dates].loc[solar_production["season"] == season].index
        average[season] = solar_production.loc[index].groupby(by="time").mean()
        average[season]["moving_avg"] = average[season][variable].rolling(5).mean()
        sns.lineplot(
            x=average[season].index,
            y=average[season]["moving_avg"],
            alpha=1,
            label=season,
            color=colors[season],
        )

    ax.legend()
    myFmt = mdates.DateFormatter('%H')
    ax.xaxis.set_major_formatter(myFmt)

    return ax


def plot_2020_demand(df, figsize=(15, 10)):
    build3 = df.loc["01-01-2020":"31-12-2020", ["Building3", "temperature"]].copy()
    build3.dropna(inplace=True)
    build3 = build3.resample(rule='7D').mean()
    build3 = build3.rolling(5).mean()
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    build3["Building3"].plot(ax=ax, kind='line', label="Consumo semanal")
    ax.set_xlim(right=dt.datetime(year=2020, month=12, day=25))

    # Summer vacation

    ax.axvspan(xmin=dt.datetime(day=16, month=3, year=2020),
               xmax=dt.datetime(day=26, month=10, year=2020),
               hatch='/', facecolor="none", edgecolor="salmon", linewidth=0.2,
               label="Cuarentena")

    ax.axvspan(xmin=dt.datetime(day=20, month=6, year=2020),
               xmax=dt.datetime(day=13, month=9, year=2020),
               hatch='///', facecolor="none", edgecolor="salmon", linewidth=0.2,
               label="5-10% Ocupación")

    ax.axvspan(xmin=dt.datetime(day=13, month=6, year=2020),
               xmax=dt.datetime(day=18, month=10, year=2020),
               hatch='//', facecolor="none", edgecolor="salmon", linewidth=0.2,
               label="25-30% Ocupación")

    ax.axvspan(xmin=dt.datetime(day=9, month=3, year=2020),
               xmax=dt.datetime(day=22, month=6, year=2020),
               color="tab:blue", linewidth=0.1, alpha=0.12, label="Periodo lectivo")

    ax.axvspan(xmin=dt.datetime(day=3, month=8, year=2020),
               xmax=dt.datetime(day=4, month=12, year=2020),
               color="tab:blue", linewidth=0.1, alpha=0.12, label=" ")
    ax.set_ylabel("Consumo energético (kW)")
    ax2 = ax.twinx()
    ax2.set_xlabel("Temperatura")
    build3["temperature"].plot(kind="line", ax=ax2, color="goldenrod", label="Temperatura semanal")
    ax.legend(loc="lower left")
    ax2.legend(loc="lower left")
    ax2.set_ylabel("Temperatura Cº")
    ax2.set_xlim(right=dt.datetime(year=2020, month=11, day=25))

    return fig

