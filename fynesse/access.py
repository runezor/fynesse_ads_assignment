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

    ## Creates the schemas
    create_pp_data_table(conn)
    create_postcode_table(conn)

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


def create_pp_data_table(conn):
    query = """"
    SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
    SET time_zone = "+00:00";

    CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
    USE `property_prices`;

    --
    -- Table structure for table `pp_data`
    --
    DROP TABLE IF EXISTS `pp_data`;
    CREATE TABLE IF NOT EXISTS `pp_data` (
      `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
      `price` int(10) unsigned NOT NULL,
      `date_of_transfer` date NOT NULL,
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
      `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
      `street` tinytext COLLATE utf8_bin NOT NULL,
      `locality` tinytext COLLATE utf8_bin NOT NULL,
      `town_city` tinytext COLLATE utf8_bin NOT NULL,
      `district` tinytext COLLATE utf8_bin NOT NULL,
      `county` tinytext COLLATE utf8_bin NOT NULL,
      `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
      `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;

    --
    -- Indexes for table `pp_data`
    --
    ALTER TABLE `pp_data`
    ADD PRIMARY KEY (`db_id`);
    MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
    CREATE INDEX `pp.postcode` USING HASH
      ON `pp_data`
        (postcode);
    CREATE INDEX `pp.date` USING HASH
      ON `pp_data` 
        (date_of_transfer);"""
    conn.cursor().execute(query)
    conn.commit()

def create_postcode_table(conn):
    query = """"USE `property_prices`;
        --
        -- Table structure for table `postcode_data`
        --
        DROP TABLE IF EXISTS `postcode_data`;
        CREATE TABLE IF NOT EXISTS `postcode_data` (
          `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
          `status` enum('live','terminated') NOT NULL,
          `usertype` enum('small', 'large') NOT NULL,
          `easting` int unsigned,
          `northing` int unsigned,
          `positional_quality_indicator` int NOT NULL,
          `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
          `lattitude` decimal(11,8) NOT NULL,
          `longitude` decimal(10,8) NOT NULL,
          `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
          `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
          `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
          `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
          `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
          `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
          `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
          `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
          `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        
        ALTER TABLE `postcode_data`
        ADD PRIMARY KEY (`db_id`);
        MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
        CREATE INDEX `po.postcode` USING HASH
          ON `postcode_data`
            (postcode);"""
    conn.cursor().execute(query)
    conn.commit()