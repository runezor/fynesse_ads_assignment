# Part-II Advanced Data Science Assignment

This is my code for the Cambridge Part-II Advanced Data Science Unit. It provides a list of functions in all levels of Fynesse which allows us to combine features from the UK Price Paid data https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads, the UK postcode data https://www.getthedata.com/open-postcode-geo and open street maps.

## Structure
This repository follows the fynesse structure (https://github.com/lawrennd/fynesse_template), meaning that it is divided into three sections: Access, assess, address.


### Access
This part is concerned with acquiring data. Make sure that you read the license part of this readme before making use of it.

It has functions that allow you to set up DB tables pp_data and postcode_data, which it will populate with data from the nationalarchives and with public UK postcode data. It also allows for rudimentary querying of open street maps.

### Assess
There's function for querying postcodes as well as pp data for a specific bounding box.

There's also a lot of queries to find data outliers as well as plotting the data on a general level.

You can use the pois_feature class along with get_pois_filtered(features, lat, lon, width, height) function to return a sanitized set of features from open street maps. Be aware if a feature is not available, it won't be part of the returned set of features.


### Address
In this phase we attempt to *address* questions, so here you'll find code that helps to answer specific questions.

Augment_with_pois takes a dataframe along with a list of computed pois features, and returns a new dataframe with extra computed columns such as min or avg distance to the pois features.

Augment_with_avg_nearby is used specifically for adding a column to a pp_data derived dataframe with the price of the 10 nearest houses.

Predict_on_data takes a fit and uses it to predict on new data.

Generate_training_and_test_indices splits data into two sets for training and testing.

Fit_on_data fits a gaussian linear model (regularized or not) on the given dataframe. The user supplies the column name for the y axis as well as the column names of features.

Get_nearest_grocery_type_for_postcode produces a dataframe with a one-hot encoding of the nearest grocery store for all the given postcodes.

Get_correlations_sorted computes the correlations of a dataframe as a flat list.

## License

### Price Paid Data
This data is licensed under the Open Government License. We have to provide a link to the license at http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/, as well as acknowledge the source. Other than that, it imposes no further restrictions, and we are free to use it for commercial use.

### Postcode data
This data is similarly licensed under the Open Government License. It was accessed via https://www.getthedata.com/open-postcode-geo.

### Open Street Maps
This data is licensed under ODbL https://www.openstreetmap.org/copyright/en. We can freely copy, distribute, transmit and adapt the data (important for our case), as long as we credit OpenStreetMap and its contributors (let this be credit!). If we use their data we must distribute under the same license, meaning we can't use it for commercial uses.


# Fynesse Template

This repo provides a template repo for doing data analysis according to the Fynesse framework.

One challenge for data science and data science processes is that they do not always accommodate the real-time and evolving nature of data science advice as required, for example in pandemic response or in managing an international supply chain. The Fynesse paradigm is inspired by experience in operational data science both in the Amazon supply chain and in the UK Covid-19 pandemic response.

The Fynesse paradigm considers three aspects to data analysis, Access, Assess, Address. 


