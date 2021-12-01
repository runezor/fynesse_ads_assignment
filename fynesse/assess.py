import datetime

from . import access
from .config import *
import math
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import mlai.plot as plot
import seaborn as sns
import pandas as pd


"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

class computed_feature:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
f
class feature:
    def __init__(self, name, is_valid, compute):
        self.name = name
        self.is_valid_for = is_valid
        self.compute = compute

    def compute_feature(self, pois):
        return computed_feature(self.name, self.compute(pois))

restaurant_feature = feature("restaurant", lambda pois: "amenity" in pois.columns.tolist(), lambda pois: pois[(pois["amenity"] == "restaurant") | (pois["amenity"] == "fast_food")])
kindergarten_feature = feature("kindergarten", lambda pois: "amenity" in pois.columns.tolist(), lambda pois: pois[pois["amenity"] == "kindergarten"])
groceries_feature = feature("groceries", lambda pois: "shop" in pois.columns.tolist(), lambda pois: pois[(pois["shop"] == "convenience") | (pois["shop"] == "supermarket")])
shop_feature = feature("shop", lambda pois: "shop" in pois.columns.tolist(), lambda pois: pois[pois["shop"].notna()])
public_transport_feature = feature("public_transport", lambda pois: "public_transport" in pois.columns.tolist(), lambda pois: pois[pois["public_transport"].notna()])
tourism_feature = feature("tourism", lambda pois: "tourism" in pois.columns.tolist(), lambda pois: pois[pois["tourism"].notna()])
hazard_feature = feature("hazard", lambda pois: "hazard" in pois.columns.tolist(), lambda pois: pois[pois["hazard"].notna()])

STANDARD_FEATURES = [restaurant_feature,kindergarten_feature,groceries_feature,shop_feature,public_transport_feature,tourism_feature,hazard_feature]


def data(credentials, database_details):
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    conn = access.data(credentials, database_details)
    return conn

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def view_pp_pricing():
    query = "choose 100000 random ones, group by flat type, year"


def view_uk_prices():
    return None
# Plot a subset of the POIs (e.g., tourist places)
# Create figure
def view_osmx_map(lat, lon, width, height):
    north, south, west, east = bounding_box(lat,lon,width=2,height=2)

    fig, ax = plt.subplots(figsize=plot.big_figsize)

    # Plot the footprint
    nodes_a = ox.geometries_from_bbox(north, south, east, west, {"amenity": True})
    nodes_b = ox.geometries_from_bbox(north, south, east, west, {"building": True})
    nodes_h = ox.geometries_from_bbox(north, south, east, west, {"historic": True})
    nodes_l = ox.geometries_from_bbox(north, south, east, west, {"leisure": True})
    nodes_s = ox.geometries_from_bbox(north, south, east, west, {"shop": True})
    nodes_t = ox.geometries_from_bbox(north, south, east, west, {"tourism": True})

    # Plot street edges
    # edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot tourist places
    nodes_a.plot(ax=ax, color="red", alpha=0.5, markersize=10)
    nodes_b.plot(ax=ax, color="green", alpha=0.5, markersize=10)
    nodes_h.plot(ax=ax, color="yellow", alpha=0.5, markersize=10)
    nodes_l.plot(ax=ax, color="blue", alpha=0.5, markersize=10)
    nodes_s.plot(ax=ax, color="orange", alpha=0.5, markersize=10)
    nodes_t.plot(ax=ax, color="brown", alpha=0.5, markersize=10)
    plt.tight_layout()

def view_pricing_for_area():
    return None

#For comparison with the above
def view_feature_for_area(data, lat, lon, w, h, feature):
    north,south,west,east = bounding_box(lat,lon, w, h)
    bins_lat = np.linspace(lat, lon, 5 + 1)
    bins_long = np.linspace(west, east, 5 + 1)
    counts, xbins, ybins = np.histogram2d(data["latitude"], data["longitude"], bins=[bins_lat, bins_long])
    sums, _, _ = np.histogram2d(data["latitude"], data["longitude"], bins=[bins_lat, bins_long], weights=data["feature"])

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 4))
    m1 = ax1.pcolormesh(ybins, xbins, counts, cmap='coolwarm')
    plt.colorbar(m1, ax=ax1)
    ax1.set_title('counts')
    m2 = ax2.pcolormesh(ybins, xbins, sums, cmap='coolwarm')
    plt.colorbar(m2, ax=ax2)
    ax2.set_title('sums')
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress possible divide-by-zero warnings
        m3 = ax3.pcolormesh(ybins, xbins, sums / counts, cmap='coolwarm')
    plt.colorbar(m3, ax=ax3)
    ax3.set_title('mean values')
    plt.tight_layout()
    plt.show()




def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    ##TODO DO the feature thing
    raise NotImplementedError

def bounding_box(lat, lon, width=2, height=2):
  height_lat = height / 110.574
  width_lon = width / (math.cos(lat*math.pi/180)*111.320)

  north = lat + height_lat / 2
  south = lat - height_lat / 2
  west = lon - width_lon / 2
  east = lon + width_lon / 2
  return north,south,west,east


def get_pp_data_for_location(conn, lat, lon, date_before, date_after, property_type, width=2, height=2):
    postcode_data_filtered = get_postcodes_in_bounding_box(lat, lon, width, height)
    postcodes = postcode_data_filtered["postcode"].values.tolist()
    if len(postcodes)<1:
        return None
    postcodes_formatted = ",".join(["'" + str(postcode) + "'" for postcode in postcodes])
    query = """select
    price as price,
    date_of_transfer as date_of_transfer,
    postcode as postcode,
    property_type as property_type,
    new_build_flag as new_build_flag,
    tenure_type as tenure_type,
    locality as locality,
    town_city as town_city,
    district as district,
    county as county
  from
    pp_data
  where
    postcode IN ({postcode_list})
  """.format(postcode_list=postcodes_formatted)
    cursor = conn.cursor()
    cursor.execute(query)
    column_names = list(map(lambda x: x[0], cursor.description))
    pp_data_for_location = pd.DataFrame(columns=column_names, data=cursor.fetchall())
    if property_type is None:
        pp_data_for_location_filter_year = pp_data_for_location[
        (pp_data_for_location["date_of_transfer"] > date_before) & (
                    pp_data_for_location["date_of_transfer"] < date_after)]
    else:
        pp_data_for_location_filter_year = pp_data_for_location[
        (pp_data_for_location["date_of_transfer"] > date_before) & (
                    pp_data_for_location["date_of_transfer"] < date_after) & (
                    pp_data_for_location["property_type"] == property_type)]

    return pp_data_for_location_filter_year

#TODO PRICE UNIT
def plot_england_price_map(conn):
    query = """select
      price,
      postcode
    FROM
      pp_data
    ORDER BY RAND ( )
    LIMIT 1000000
    """
    postcode_positions = access.get_postcode_positions()
    cursor = conn.cursor()
    cursor.execute(query)
    column_names = list(map(lambda x: x[0], cursor.description))
    prices = pd.DataFrame(columns=column_names, data=cursor.fetchall())
    prices_with_location = postcode_positions.merge(prices)

    x_cut = pd.cut(prices_with_location.longitude, np.linspace(-7, 2.5, 48), right=False)
    y_cut = pd.cut(prices_with_location.latitude, np.linspace(49, 57, 70), right=False)
    prices = prices_with_location.groupby([x_cut, y_cut]).mean().reset_index()

    indices = prices["longitude"].unique()
    cols = prices["latitude"].unique()
    df = pd.DataFrame(np.array(prices["price"].tolist()).reshape(48 - 1, 70 - 1), index=indices, columns=cols)

    ax = sns.heatmap(df.T)
    ax.invert_yaxis()

def plot_quotients(fit, names):
    fig, ax = plt.subplots(figsize=(16, 4))
    plt.xticks(rotation='vertical')

    ax.bar(names + ["Constant"], [x for x in fit.params.tolist()], 0.35, label='Quotient')

    ax.set_ylabel('Quotient')
    ax.set_title('value of quotient for features')
    ax.legend()
    plt.show()

#Actual should be a list or numpy array
#Predictions should be a pandas frame with columns mean, obs_ci_upper and obs_ci_lower
def plot_predictions_vs_actual(predictions,actual,x_label,y_label,title):
    #TODO R VALUE
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    max_val = max(max(predictions["mean"]),max(actual))*1.1
    min_val = min(min(predictions["mean"]),min(actual))*0.9
    ax.set_xlim([min_val,max_val])
    ax.set_ylim([min_val,max_val])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    upper_error=(predictions["obs_ci_upper"]-predictions["mean"]).to_numpy()
    lower_error=(predictions["mean"]-predictions["obs_ci_lower"]).to_numpy()
    ax.errorbar(actual,predictions["mean"].to_numpy(),fmt='o', yerr=(lower_error,upper_error))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()


def plot_pois(lat, lon, title, w=2, h=2):
    # Plot a subsets of the POIs
    fig, ax = plt.subplots(figsize=plot.big_figsize)

    tags = ["amenity", "buildings", "historic", "leisure", "shop", "tourism"]
    north, south, west, east = bounding_box(lat, lon, width=w, height=h)
    nodes = ox.geometries_from_bbox(north, south, east, west, {i: True for i in tags})

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    types = ["amenity", "building", "historic", "leisure", "shop", "tourism"]
    colors = ["red", "green", "yellow", "blue", "orange", "brown"]

    # Plot tourist places
    for tag, color in zip(types, colors):
        if tag in nodes.columns:
            nodes[nodes[tag].notna()].plot(ax=ax, color=color, alpha=0.5, markersize=10)
    ax.set_title(title)
    plt.tight_layout()


def plot_features_in_grid(data, features, lat, lon, w, h, size):
    north, south, west, east = bounding_box(lat, lon, width=w, height=h)
    x_cut = pd.cut(data.longitude, np.linspace(west, east, size), right=False)
    y_cut = pd.cut(data.latitude, np.linspace(south, north, size), right=False)
    data_grouped = data.groupby([x_cut, y_cut]).mean().reset_index()

    if len(features) > 1:
        rows = int((len(features) + 3) / 4)
        f, axi = plt.subplots(ncols=4, nrows=rows)
        f.tight_layout()
        f.set_figwidth(17)
        f.set_figheight(rows * 4)
        for feature, ax in zip(features, axi.flatten()[0:len(features)]):
            ax.set_title(feature)
            data_mod = pd.DataFrame(np.array(data_grouped[feature].to_list()).reshape(size - 1, size - 1),
                                    index=data_grouped["longitude"].unique(), columns=data_grouped["latitude"].unique())
            sns.heatmap(data_mod.T, ax=ax)
    else:
        f, ax = plt.subplots()
        ax.set_title(features[0])
        data_mod = pd.DataFrame(np.array(data_grouped[features[0]].to_list()).reshape(size - 1, size - 1),
                                index=data_grouped["longitude"].unique(), columns=data_grouped["latitude"].unique())
        sns.heatmap(data_mod.T, ax=ax)

def plot_prices_over_year(pp_data, ax):
  prices_over_year = pp_data.groupby("year(date_of_transfer)").mean()
  ax.set_xlabel("year")
  ax.set_ylabel("price (£)")
  ax.set_title("Average house price")
  ax.plot(prices_over_year.index,prices_over_year["price"])

def plot_pp_data_proportions(pp_data, ax):
  property_proportions = (pp_data.groupby("property_type").count()/1000000)*100

  ax.pie(property_proportions["price"], labels=property_proportions.index, autopct='%1.1f%%',
          shadow=True, startangle=90)
  ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  ax.set_title("Proportion of different property types in dataset")

def plot_prices_by_year_new(pp_data, ax):
  prices_by_year_build = pp_data.groupby(["new_build_flag","year(date_of_transfer)"]).mean()
  ax.plot(prices_by_year_build.xs("Y").index,prices_by_year_build.xs("Y")["price"])
  ax.plot(prices_by_year_build.xs("N").index,prices_by_year_build.xs("N")["price"])
  ax.set_xlabel("year")
  ax.set_ylabel("price (£)")
  ax.set_title("Average price of New/Old houses")
  ax.legend(["New", "Old"])

def plot_prices_by_year_type(pp_data, ax):
  prices_by_type = pp_data.groupby(["property_type","year(date_of_transfer)"]).mean()

  types = ["Detached", "Flat", "Other", "Terraced", "Semidetached"]
  for prop_type in types:
    initial = prop_type[0:1]
    plt.plot(prices_by_type.xs(initial).index,prices_by_type.xs(initial)["price"])
  ax.legend(types)
  ax.set_xlabel("year")
  ax.set_ylabel("price (£)")
  ax.set_title("Average price of different house types")

def print_pp_data_outliers(conn):
  ## Find number of outliers
  query = """select
    *
  FROM
    pp_data
  WHERE
    price<1000 or
    year(date_of_transfer)<1995 or 
    year(date_of_transfer)>2021
  """

  cursor = conn.cursor()
  cursor.execute(query)
  column_names = list(map(lambda x: x[0], cursor.description))
  df = pd.DataFrame(columns=column_names, data=cursor.fetchall())
  print("Prices less than 1000")
  index = df["price"]<1000
  print(df[index]["price"])
  print("Outside years")
  index = (df["date_of_transfer"]<datetime.date(1995,1,1)) | (df["date_of_transfer"]>datetime.date(2021,12,1))
  print(df[index]["date_of_transfer"])


def get_postcodes_in_bounding_box(lat,lon, width, height):
  if access.get_postcode_positions() is None:
      raise("Postcodes not cached yet")
  north, south, west, east = bounding_box(lat,lon,width=width,height=height)
  return access.get_postcode_positions()[(access.POSTCODE_POSITIONS["longitude"]<east) & (west<access.POSTCODE_POSITIONS["longitude"]) & (access.POSTCODE_POSITIONS["latitude"]<north) & (south<access.POSTCODE_POSITIONS["latitude"])]


def lat_lon_to_dist(lat_diff, lon_diff, lat):
  height = 110.574 * lat_diff
  width = 111.320 * math.cos(lat*math.pi/180)*lon_diff

  return math.sqrt((height)**2+(width)**2)
