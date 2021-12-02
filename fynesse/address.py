# This file contains code for suporting addressing questions in the data

import decimal
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy.special import factorial
from scipy.stats import norm
import datetime

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

# Used for describing a bounding box + a searching radius
class search_area:
  def __init__(self, w, h, r):
    self.w = w
    self.h = h
    self.max_search_r = r

# Returns a pp_data subset in the given area, flexible to accommodate different data densitites
# :param conn: DB connection
# :param latitude: latitude of area
# :param longitude: longitude of area
# :param data: Date for filtering
# :param property_type: Property type for filtering
# :return pp_data, a search_area, as well as a list of tradeoffs
def find_pp_data_flexible(conn, latitude, longitude, date, property_type):
    # Constants, may need tuning
    MAX_R = 0.5
    w = 1
    h = 1
    MIN_PP_DATA = 100
    TIME_DELTA = datetime.timedelta(365)
    tradeoffs = []

    if property_type is "O":
        tradeoffs += ["Warning: Fits may not generalize well for type 'Other', due to great variation in the data."]

    pp_in_area = assess.get_pp_data_for_location(conn, latitude, longitude, date - TIME_DELTA, date + TIME_DELTA,
                                                 property_type, width=w, height=h)

    # Reject queries if data is bad, since this is easy to detect. Handling of bad lat/lon is done below
    if date + TIME_DELTA < datetime.date(1995, 1, 1) or date - TIME_DELTA > datetime.date(2021, 12, 1):
        print("Sorry, request is outside time boundary, and we don't extrapolate *yet*")
        raise

    # Expand bounding box if we don't get enough data
    n = 0
    # TRADEOFF 1: Double the bounding box
    while ((pp_in_area is None or len(pp_in_area.index) < MIN_PP_DATA) and n < 4):
        n = n + 1
        w = w * 2
        h = h * 2
        MAX_R = MAX_R * 2
        tradeoffs += ["Had to double bounding box"]
        pp_in_area = assess.get_pp_data_for_location(conn, latitude, longitude, date - TIME_DELTA, date + TIME_DELTA,
                                                     property_type, width=w, height=h)

    # TRADEOFF 2: Double time delta
    if (pp_in_area is None or len(pp_in_area.index) < MIN_PP_DATA):
        pp_in_area = assess.get_pp_data_for_location(conn, latitude, longitude, date - (TIME_DELTA + TIME_DELTA),
                                                     date + (TIME_DELTA + TIME_DELTA), property_type, width=w, height=h)
        tradeoffs += ["Had to double time delta"]

    # TRADEOFF 3: Eliminate property_type filter
    if (pp_in_area is None or len(pp_in_area.index) < MIN_PP_DATA):
        pp_in_area = assess.get_pp_data_for_location(conn, latitude, longitude, date - (TIME_DELTA + TIME_DELTA),
                                                     date + (TIME_DELTA + TIME_DELTA), None, width=w, height=h)
        tradeoffs += ["Had to eliminate property_type filter"]

    # TRADEOFF 4: Give up
    if (pp_in_area is None):
        print("Sorry, no postcodes available")
        raise
    if (len(pp_in_area.index) < MIN_PP_DATA):
        print("Sorry, too few sales in area / time. I got " + str(len(pp_in_area.index)) + " sales")
        raise

    return pp_in_area, search_area(w, h, MAX_R), tradeoffs

# Prints info about a prediction
# :param prediction: A statsmodels prediction
# :param area: a search_area
# :param fit: A fit
# :param tradeoffs: List of tradeoffs
# :return Prints as a side effect
def print_prediction(prediction, area, fit, tradeoffs):
    print("Using bounding box of size: " + str(area.w) + "*" + str(area.h))
    print("Mean of prediction dist. is: " + str(prediction["mean"]))
    print("Upper ci of prediction dist. is: " + str(prediction["mean_ci_upper"]))
    print("Lower ci of prediction dist. is: " + str(prediction["mean_ci_lower"]))
    print("R^2 of fit is: " + str(fit.rsquared))
    if len(tradeoffs) > 0:
        print("Tradeoffs are...")
        for tradeoff in tradeoffs:
            print(tradeoff)

        print(
            "Tradeoffs make for a less precise model, since we're fitting on more general data that may not reflect the queried data well")


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

    return sorted_correlations

# Plots the fit of on a dataset using PCA-acquired features of different dimensions
# :param priors: A list of priors
# :param data: A pandas dataframe
# :return bool(p<0.05)
def is_confidence_in_input_low(priors, data):

  def poisson(k, lambd):
    return(lambd**k/factorial(k))*np.exp(-lambd)

  p = 1
  for prior in priors:
    val = data[prior["column"]]
    if prior["type"] is "poisson":
      p_sub = poisson(val, prior["params"])
    elif prior["type"] is "normal":
      # We need a precision of val to calculate this accurately
      # Given that we haven't built in a prediction interval for our features, I've had to estimate it
      # This leads to a bad calculation, but hopefully it gives an idea of what a more sophisticated model could do
      # We set the uncertainty to 100m
      eps = 0.1
      p_sub = norm(prior["params"][0], prior["params"][1]).cdf(val+eps)-norm(prior["params"][0], prior["params"][1]).cdf(val-eps)
    elif prior["type"] is "zero-val":
      if val==0:
        p_sub = 1
      else:
        p_sub = 0
    p *= p_sub

  return p<0.05

# Plots the fit of on a dataset using PCA-acquired features of different dimensions
# :param data: A dataframe containing at least columns 'latitude' and 'longitude'
# :param fit_on: Column that we fit on
# :param feature_cols: Columns that we fit with
# :param normalized: Whether or not we normalize features
# :return plot as a side effect
def plot_PCA_feature_reduction(data, fit_on, feature_cols, normalized = False):
  X = data[feature_cols]
  Y = data[fit_on]

  if normalized:
    X=(X-X.mean())/X.std()

  fit_vals = []
  for feature_count in range(1,len(feature_cols)):
    pca = PCA(n_components=feature_count)
    pca.fit(X)
    feature_names = [str(i) for i in range(0,feature_count)]
    new_features = pd.DataFrame(pca.transform(X), columns=feature_names)
    new_features[fit_on] = Y

    #Do fitting on the training data
    fit = address.fit_on_data(new_features, fit_on, feature_names)

    #Do predictions for testing data
    predictions = address.predict_on_data(fit, new_features, feature_names)

    fit_vals.append(fit.rsquared)
  plt.plot(range(1,len(feature_cols)), fit_vals)
  plt.xticks(range(1,len(feature_cols)))
  plt.xlabel("Features")
  plt.ylabel("R squared")
  plt.title("Change in R squared for different feature counts")