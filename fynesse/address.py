# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""
import decimal
import pandas as pd
import numpy as np
import statsmodels.api as sm

from . import access
from . import assess

"""Address a particular question that arises from the data"""


class computed_feature:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows

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

## In: DF must have latitude + longitude
## Out: Same df, but with new feature columns
def augment_df_with_pois(df, features, MAX_RADIUS):
    feature_types = ["min", "avg_dist", "within_r"]

    data = []
    i = 0
    for index, row in df.iterrows():
        if i%10==0:
            print(str(i) + "/" + str(len(df.index)))
        i = i + 1
        latitude, longitude = row["latitude"], row["longitude"]
        joined_row = []
        for feature in features:
            min_dist = MAX_RADIUS
            total_dist = 0
            within_r = 0
            n = 0
            for index, row in feature.rows.iterrows():
                diff_x = (decimal.Decimal(row["geometry"].centroid.x) - longitude)
                diff_y = (decimal.Decimal(row["geometry"].centroid.y) - latitude)
                dist = assess.lat_lon_to_dist(float(diff_y), float(diff_x), float(latitude))

                if dist < MAX_RADIUS:
                    within_r += 1
                    n += 1
                    total_dist += dist

                if dist < min_dist:
                    min_dist = dist

            # Generate rows
            joined_row.append(min_dist)
            joined_row.append(MAX_RADIUS if n == 0 else total_dist / n)
            joined_row.append(within_r)
        data.append(joined_row)

    # Generate column names
    columns = []
    for feature in features:
        for feature_type in feature_types:
            columns.append(feature.name + "_" + feature_type)

    feature_df = pd.DataFrame(data, columns=columns, index=df.index)

    return feature_df, columns


def predict_on_data(basis, data, feature_column_titles, regularized=False):
    feature_columns = [np.array(data[x], dtype=float).reshape(-1, 1) for x in feature_column_titles] + [
        np.ones(len(data)).reshape(-1, 1)]

    design = np.concatenate(feature_columns, axis=1)
    if regularized:
        y_predictions = basis.predict(design).summary_frame(alpha=0.05)
    else:
        y_predictions = basis.get_prediction(design).summary_frame(alpha=0.05)
    return y_predictions

def generate_training_and_test_indices(data, ratio=0.8):
    size = len(data.index)
    training_data = np.random.permutation(
        np.append(np.ones(int(size * ratio), dtype=bool), (np.zeros(size - int(size * 0.8), dtype=bool))))
    test_data = (np.ones(size) - training_data).astype(bool)

    return data[training_data], data[test_data]

def fit_on_data(data, y_column, feature_column_titles, regularize = False):
  y = data[y_column]

  feature_columns = [np.array(data[x], dtype=float).reshape(-1,1) for x in feature_column_titles]+[np.ones(len(data)).reshape(-1,1)]

  design = np.concatenate(feature_columns,axis=1)

  m_linear_basis = sm.OLS(y,design)

  if regularize:
    results_basis = m_linear_basis.fit_regularized(L1_wt=0.1)
  else:
    results_basis = m_linear_basis.fit()

  return results_basis


def get_nearest_grocery_type_for_postcode(conn, lat, lon, width=2, height=2, MAX_RADIUS=0.5):
    data = []
    postcode_pos = assess.get_postcodes_in_bounding_box(lat, lon, width, height)
    pois = access.get_pois(lat, lon, tags={"shop": True}, width=2 + MAX_RADIUS, height=2 + MAX_RADIUS)

    pois_shops = pois[(pois["shop"].notna()) & (pois["brand"].notna())]

    supermarkets = ["waitrose", "marks and spencers", "sainsburys", "morrisons", "tesco", "co-op", "asda", "aldi",
                    "lidl", "iceland"]
    supermarket_synonyms = [["waitrose"],
                            ["marks and spencer", "m&s", "mark's and spencer"],
                            ["sainsburys", "sainsbury's"],
                            ["morrisons"],
                            ["tesco"],
                            ["co-op", "coop"],
                            ["asda"],
                            ["aldi"],
                            ["lidl"],
                            ["iceland"]]

    def get_shop_name(s):
        for synonyms in supermarket_synonyms:
            for synonym in synonyms:
                if synonym in s.lower():
                    return synonyms[0]
        return None

    ## Which shops are available
    included_supermarkets = []
    for synonym_set in supermarket_synonyms:
        included = False
        for index, row in pois_shops.iterrows():
            for synonym in synonym_set:
                if synonym in row["brand"].lower():
                    included = True
        if included:
            included_supermarkets += [synonym_set[0]]

    i = 0
    for index, row in postcode_pos.iterrows():
        i = i + 1
        postcode, latitude, longitude = row["postcode"], row["latitude"], row["longitude"]
        joined_row = [postcode]

        closest_dist = MAX_RADIUS
        closest_shop = None
        # Todo could be optimised somewhat
        for index2, shop in pois_shops.iterrows():
            diff_x = (decimal.Decimal(shop["geometry"].centroid.x) - longitude)
            diff_y = (decimal.Decimal(shop["geometry"].centroid.y) - latitude)
            dist = assess.lat_lon_to_dist(float(diff_y), float(diff_x), float(latitude))

            if dist < closest_dist and get_shop_name(shop["brand"]) is not None:
                closest_shop = get_shop_name(shop["brand"])
                closest_dist = dist

        # One hot encoding
        for supermarket in included_supermarkets:
            if closest_shop is None or not get_shop_name(closest_shop) == supermarket:
                joined_row += [0]
            else:
                joined_row += [1]
        data.append(joined_row)

    columns = []

    shop_column_names = [x + "_is_nearest" for x in included_supermarkets]
    postcodes_with_nearest_supermarket = pd.DataFrame(data, columns=["postcode"] + shop_column_names)

    return postcodes_with_nearest_supermarket, shop_column_names