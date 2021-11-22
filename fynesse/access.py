from .config import *
import pymysql
import urllib.request

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data(credentials, database_details):
    """Read the data from the web or local file, returning structured format such as a data frame"""

    ## Get DB access
    conn = create_connection(user=credentials["username"],
                             password=credentials["password"],
                             host=database_details["url"],
                             database="property_prices")

    ## Uploads the data to the SQL server
    load_pp_to_sql(conn)
    load_postcode_to_sql(conn)

    return conn



def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def load_pp_to_sql(conn):
    parts = ["1", "2"]
    years = ["2018", "2017"]


    query = """LOAD DATA LOCAL INFILE 'temp.csv' INTO TABLE `pp_data` FIELDS TERMINATED BY ',' LINES STARTING BY '' TERMINATED BY '\n';"""

    for year in years:
      for part in parts:
        filename = 'pp-'+year+'-part'+part+'.csv'
        urllib.request.urlretrieve('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/'+filename, 'temp.csv')
        conn.cursor().execute(query)
        conn.commit()

def load_postcode_to_sql(conn):
    urllib.request.urlretrieve('https://www.getthedata.com/downloads/open_postcode_geo.csv.zip', 'temp.csv')

    query = """LOAD DATA LOCAL INFILE 'temp.csv' INTO TABLE `postcode_data`
    FIELDS TERMINATED BY ',' 
    LINES STARTING BY '' TERMINATED BY '\n';"""

    conn.cursor().execute(query)
    conn.commit()

