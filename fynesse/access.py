from .config import *
import pymysql
import urllib.request
import pandas as pd
import shutil
import osmnx as ox
from . import assess

import math

# This file accesses the data

# Cache for postcodes
global POSTCODE_POSITIONS
POSTCODE_POSITIONS = None


#Builds the DB
#Takes a set of credentials + database details
# :param credentials: dict containing a username and password
# :param database_details: dict containing a url for the database
# :warning: Will clear tables in DB if overlapping
def build(credentials, database_details):
    conn = create_connection(user=credentials["username"],
                             password=credentials["password"],
                             host=database_details["url"],
                             database="property_prices")

    create_postcode_table(conn)

    ## Uploads the data to the SQL server
    load_pp_to_sql(conn)
    load_postcode_to_sql(conn)

#Establishes a connection to the DB, and fills out the postcode cachce
# :param credentials: dict containing a username and password
# :param database_details: dict containing a url for the database
# :return: Connection to db
def data(credentials, database_details):
    """Read the data from the web or local file, returning structured format such as a data frame"""

    ## Get DB access
    conn = create_connection(user=credentials["username"],
                             password=credentials["password"],
                             host=database_details["url"],
                             database="property_prices")

    #Caches postcodes
    get_postcode_positions(conn=conn)

    return conn

# Fetches postcode locations from the DB if not cached
# :param conn: Connection to DB, only necessary if data is not cached
# :return: Postcodes dataframe
def get_postcode_positions(conn=None):
    global POSTCODE_POSITIONS

    if POSTCODE_POSITIONS is not None:
        return POSTCODE_POSITIONS
    if conn is None:
        return None
    query = """select postcode, latitude, longitude
        from postcode_data
    """

    cursor = conn.cursor()
    cursor.execute(query)
    column_names = list(map(lambda x: x[0], cursor.description))
    postcode_data = pd.DataFrame(columns=column_names, data=cursor.fetchall())
    POSTCODE_POSITIONS = postcode_data
    return postcode_data

# Creates a pymysql connection to the DB
# :param user: The username
# :param password: The user password
# :param host: The host url
# :param database: The database url
# :param port: The port number
# :return: Connection to the DB
def create_connection(user, password, host, database, port=3306):
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

# Loads price point data into the DB
# :param conn: the connection to the DB
def load_pp_to_sql(conn):
    parts = ["1", "2"]
    years = [str(i) for i in range(1995, 2022)]

    query = """LOAD DATA LOCAL INFILE 'temp.csv'
    INTO TABLE `pp_data`
    FIELDS OPTIONALLY ENCLOSED BY '"'
    TERMINATED BY ','
    LINES STARTING BY ''
    TERMINATED BY '\n';"""

    for year in years:
        for part in parts:
            filename = 'pp-' + year + '-part' + part + '.csv'
            print("Retrieving data...")
            urllib.request.urlretrieve(
                'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/' + filename, 'temp.csv')
            print("Uploading...")
            conn.cursor().execute(query)
            conn.commit()
            print("Uploaded " + year + " part " + part)

# Loads postcode data into the database
# :param conn: the connection to the DB
def load_postcode_to_sql(conn):
    urllib.request.urlretrieve('https://www.getthedata.com/downloads/open_postcode_geo.csv.zip', 'temp.csv.zip')

    shutil.unpack_archive("temp.csv.zip", 'zip/')

    query = """LOAD DATA LOCAL INFILE 'zip/open_postcode_geo.csv' INTO TABLE `postcode_data`
    FIELDS TERMINATED BY ',' 
    LINES TERMINATED BY '\n';"""

    conn.cursor().execute(query)
    conn.commit()


# Prepares the pricepoint table in the DB
# :warning Will drop the table if it already exists
# :param conn: the connection to the DB
def create_pp_data_table(conn):
    drop_table_query = "DROP TABLE IF EXISTS pp_data"
    create_table_query = """CREATE TABLE pp_data (
          transaction_unique_identifier tinytext COLLATE utf8_bin NOT NULL,
          price int(10) unsigned NOT NULL,
          date_of_transfer date NOT NULL,
          postcode varchar(8) COLLATE utf8_bin NOT NULL,
          property_type varchar(1) COLLATE utf8_bin NOT NULL,
          new_build_flag varchar(1) COLLATE utf8_bin NOT NULL,
          tenure_type varchar(1) COLLATE utf8_bin NOT NULL,
          primary_addressable_object_name tinytext COLLATE utf8_bin NOT NULL,
          secondary_addressable_object_name tinytext COLLATE utf8_bin NOT NULL,
          street tinytext COLLATE utf8_bin NOT NULL,
          locality tinytext COLLATE utf8_bin NOT NULL,
          town_city tinytext COLLATE utf8_bin NOT NULL,
          district tinytext COLLATE utf8_bin NOT NULL,
          county tinytext COLLATE utf8_bin NOT NULL,
          pd_category_type varchar(2) COLLATE utf8_bin NOT NULL,
          record_status varchar(2) COLLATE utf8_bin NOT NULL,
          db_id bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET = utf8 COLLATE = utf8_bin AUTO_INCREMENT = 1;
        """

    create_index_pc_query = """CREATE INDEX pp_postcode USING HASH ON pp_data (postcode)"""
    create_index_date_query = """CREATE INDEX pp_date USING HASH ON pp_data (date_of_transfer)"""
    create_index_type_query = """CREATE INDEX pp_type USING HASH ON pp_data (property_type)"""

    conn.cursor().execute(drop_table_query)
    conn.cursor().execute(create_table_query)
    conn.commit()
    conn.cursor().execute(create_index_pc_query)
    conn.cursor().execute(create_index_date_query)
    conn.cursor().execute(create_index_type_query)
    conn.commit()

# Prepares the postcode table
# :warning Will drop the table if it already exists
# :param conn: the connection to the DB
def create_postcode_table(conn):
    drop_table_query = "DROP TABLE IF EXISTS postcode_data"
    create_table_query = """CREATE TABLE postcode_data (
          postcode varchar(8) COLLATE utf8_bin NOT NULL,
          status enum('live','terminated') NOT NULL,
          usertype enum('small', 'large') NOT NULL,
          easting int unsigned,
          northing int unsigned,
          positional_quality_indicator int NOT NULL,
          country enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
          latitude decimal(11,8) NOT NULL,
          longitude decimal(10,8) NOT NULL,
          postcode_no_space tinytext COLLATE utf8_bin NOT NULL,
          postcode_fixed_width_seven varchar(7) COLLATE utf8_bin NOT NULL,
          postcode_fixed_width_eight varchar(8) COLLATE utf8_bin NOT NULL,
          postcode_area varchar(2) COLLATE utf8_bin NOT NULL,
          postcode_district varchar(4) COLLATE utf8_bin NOT NULL,
          postcode_sector varchar(6) COLLATE utf8_bin NOT NULL,
          outcode varchar(4) COLLATE utf8_bin NOT NULL,
          incode varchar(3)  COLLATE utf8_bin NOT NULL,
          db_id bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin"""

    add_db_id_field_query = """ALTER TABLE postcode_data ADD PRIMARY KEY (db_id)""";
    add_db_autoinc_field_query = """ALTER TABLE postcode_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1""";

    create_index_query = """CREATE INDEX `po.postcode` USING HASH
          ON `postcode_data`
            (postcode);"""
    create_index_lon_query = """CREATE INDEX postcode_lon USING HASH ON postcode_data (longitude)"""
    create_index_lat_query = """CREATE INDEX postcode_lat USING HASH ON postcode_data (latitude)"""

    conn.cursor().execute(drop_table_query)
    conn.cursor().execute(create_table_query)
    conn.cursor().execute(add_db_id_field_query)
    conn.cursor().execute(add_db_autoinc_field_query)
    conn.cursor().execute(create_index_query)
    conn.cursor().execute(create_index_lon_query)
    conn.cursor().execute(create_index_lat_query)
    conn.commit()

# Gets points of interests from OSMNX
# :param lat: The latitude of the box center
# :param lon: The longitude of the box center
# :param width: The width of the box in km
# :param height: The height of the box in km
# :param tags: Keys for OSMNX
# :warning The data here is unfiltered, so care is needed when handling the result, ideally use assess get_pois_filtered
def get_pois(lat, lon, tags={"amenity": True}, width=2, height=2):
  north, south, west, east = assess.bounding_box(lat,lon,width=width,height=height)

  return ox.geometries_from_bbox(north, south, east, west, tags)