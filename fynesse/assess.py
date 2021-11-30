import datetime

from . import access
from .config import *
import math
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import mlai
import mlai.plot as plot
import pandas as pd


"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""
def data(credentials, database_details):
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    conn = access.data(credentials, database_details)


    query = """select postcode from pp_data where postcode<>'""' limit 100"""

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()

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
    pp_data_for_location_filter_year = pp_data_for_location[
        (pp_data_for_location["date_of_transfer"] > date_before) & (
                    pp_data_for_location["date_of_transfer"] < date_after) & (
                    pp_data_for_location["property_type"] == property_type)]

    return pp_data_for_location_filter_year

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
    ax.set_xlim([0,1000000])
    ax.set_ylim([0,1000000])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    upper_error=(predictions["obs_ci_upper"]-predictions["mean"]).to_numpy()
    lower_error=(predictions["mean"]-predictions["obs_ci_lower"]).to_numpy()
    ax.errorbar(actual,predictions["mean"].to_numpy(),fmt='o', yerr=(lower_error,upper_error))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()

def get_postcodes_in_bounding_box(lat,lon, width, height):
  if access.get_postcode_positions() is None:
      raise("Postcodes not cached yet")
  north, south, west, east = bounding_box(lat,lon,width=width,height=height)
  return access.get_postcode_positions()[(access.POSTCODE_POSITIONS["longitude"]<east) & (west<access.POSTCODE_POSITIONS["longitude"]) & (access.POSTCODE_POSITIONS["latitude"]<north) & (south<access.POSTCODE_POSITIONS["latitude"])]


def lat_lon_to_dist(lat_diff, lon_diff, lat):
  height = 110.574 * lat_diff
  width = 111.320 * math.cos(lat*math.pi/180)*lon_diff

  return math.sqrt((height)**2+(width)**2)
