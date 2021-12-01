# This file contains code for suporting addressing questions in the data

import decimal
import pandas as pd
import numpy as np
import statsmodels.api as sm

from . import access
from . import assess


# Write a bunch of latitude/longitude positions for ease of use
class position:
    def __init__(self, name, lat, lon):
        self.name = name
        self.latitude = lat
        self.longitude = lon

    def position(self):
        return self.latitude, self.longitude

BIRMINGHAM = position("Birmingham", 52.489471, -1.898575)
LONDON = position("London", 51.509865, -0.118092)
NEWCASTLE = position("Newcastle", 54.966667, -1.6)
CAMBRIDGE = position("Cambridge", 52.205276, 0.119167)
EDINBURGH = position("Edinburgh", 55.953251, -3.188267)
ST_IVES = position("St. Ives", 52.33203, -0.07612)


# Returns a sanitised set of POIS based on a feature set
# :param df: A dataframe containing at least columns 'latitude' and 'longitude'
# :param features: A list of pois_computed_feature
# :param MAX_RADIUS: The maximal radius for average and minimum distance calculations
# :param debug: Set true for progress output
# :return A dataframe matching the index of df, a list of column names for the features
def augment_df_with_pois(df, features, MAX_RADIUS, debug=False):
    feature_types = ["min", "avg_dist", "within_r"]

    data = []
    i = 0
    for index, row in df.iterrows():
        if debug and i % 10 == 0:
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


# Returns a dataframe with an added column "avg_nearby", which averages the price of the 10 nearest houses
# :param pp_with_locations: A dataframe containing pricepoint data along with 'latitude' and 'longitude'
# :param n: How many houses to average over
# :return A dataframe copy of pp_with_locations, but with an additional column "avg_nearby"
def augment_with_avg_nearby_price(pp_with_locations, n=10):
    out = pp_with_locations.copy()

    def get_avg_price_of_nearby(lat, lon):
        prices = []
        for index, row in pp_with_locations.iterrows():
            diff_x = row["longitude"] - lon
            diff_y = row["latitude"] - lat
            dist = assess.lat_lon_to_dist(float(diff_y), float(diff_x), float(lat))

            prices.append({"price": row["price"], "dist": dist})
            if len(prices) > n ** 2:
                for price in prices:
                    prices = sorted(prices, key=lambda d: d['dist'])
                    prices = prices[0:(n - 1)]

        prices = sorted(prices, key=lambda d: d['dist'])
        prices = prices[0:(n - 1)]

        sum = 0
        for price in prices:
            sum += price["price"]

        avg = sum / len(prices)

        return avg

    out["avg_nearby"] = pp_with_locations.apply(lambda row: get_avg_price_of_nearby(row["latitude"], row["longitude"]),
                                                axis=1)
    return out


# Predicts on data assuming (statsmodels.ols)
# :param basis: The fit
# :param data: The data to fit on, should be a pandas dataframe
# :param feature_column_titles: The names of the feature columns
# :param regularized: Whether or not the fit is regularized (to handle statsmodels weirdness)
# :return A list of predicitons
def predict_on_data(basis, data, feature_column_titles, regularized=False):
    feature_columns = [np.array(data[x], dtype=float).reshape(-1, 1) for x in feature_column_titles] + [
        np.ones(len(data)).reshape(-1, 1)]

    design = np.concatenate(feature_columns, axis=1)
    if regularized:
        y_predictions = basis.predict(design).summary_frame(alpha=0.05)
    else:
        y_predictions = basis.get_prediction(design).summary_frame(alpha=0.05)
    return y_predictions


# Divides data into two sets, handy for generatingtraining and test data
# :param data: The data to divide
# :param ratio: The ratio of training_data/data, the rest goes in test_data
# :return A tuple with training and test data
def generate_training_and_test_indices(data, ratio=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    size = len(data.index)
    training_data = np.random.permutation(
        np.append(np.ones(int(size * ratio), dtype=bool), (np.zeros(size - int(size * 0.8), dtype=bool))))
    test_data = (np.ones(size) - training_data).astype(bool)

    return data[training_data], data[test_data]


# Fits on data using statsmodels.ols
# :param data: The data to fit on, should be a pandas dataframe
# :param y_column: The name of the y column, the one we fit to
# :param feature_column_titles: A list containing the names of the columns we'd like to use as features
# :param regularize: Whether or not we should use L2
# :return A list of predicitons
def fit_on_data(data, y_column, feature_column_titles, regularize=False):
    y = data[y_column]

    feature_columns = [np.array(data[x], dtype=float).reshape(-1, 1) for x in feature_column_titles] + [
        np.ones(len(data)).reshape(-1, 1)]

    design = np.concatenate(feature_columns, axis=1)

    m_linear_basis = sm.OLS(y, design)

    if regularize:
        results_basis = m_linear_basis.fit_regularized(L1_wt=0, alpha=0.01)
    else:
        results_basis = m_linear_basis.fit()

    return results_basis


# Computes the nearest grocery stores for each postcode in a bounding box
# :param lat: The latitude of the box center
# :param lon: The longitude of the box center
# :param width: The width of the box in km
# :param height: The height of the box in km
# :param MAX_RADIUS: The maximal radius we will go looking for a grocery store
# :return A dataframe with postcodes, along with a one_hot encoding of the nearest grocery store
def get_nearest_grocery_type_for_postcode(lat, lon, width=2, height=2, MAX_RADIUS=1):
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


# Computes the correlations of a dataframe as a flat list
# :param df: The dataframe
# :return A list of correlations with of the format {"name": 'variable1/variable', "correlation": cor(variable1,variable2)}
def get_correlations_sorted(df):
    df_c = df.corr()
    correlations = []
    for index, row in df_c.iterrows():
        for column in df_c.columns:
            if index is not column:
                correlations += [{"name": index + "/" + column, "correlation": abs(row[column])}]

    sorted_correlations = sorted(correlations, key=lambda d: d['correlation'])

    return df_c
