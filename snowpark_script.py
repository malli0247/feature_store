import warnings
import yaml
import json
import pandas as pd
from typing import List, Dict, Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.functions import col
from snowflake.ml.modeling.preprocessing import OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer

from snowflake.ml.feature_store import FeatureStore, FeatureView, Entity, CreationMode

warnings.simplefilter(action="ignore", category=UserWarning)


def snowpark_session(connection_parameters):
    '''
    Initialize snowpark session to connect with a Snowflake databaseIncase, 
    Snowflake configured with single sign-on (SSO), configure 
    your client application to use browser-based SSO for authentication.

    connection_parameters : 
    '''
    if "password" not in connection_parameters and "authenticator" not in connection_parameters:
        connection_parameters["authenticator"] = "externalbrowser"
    try:
        session = Session.builder.configs(connection_parameters).create()
        print(f"session:", {session})

    except Exception as e:
        print(f"Error while creating session: {e}")
        raise e

    return session


def load_data(session: snowpark.Session, database: str, schema: str, source_table: str) -> DataFrame:
    """
    Loads source tables from respecitve database, schema from active snowpark session
    
    session      : Snowpark session
    database     : Snowflake database to use
    schema       : Snowflake schema to use
    source_table : Source table to use

    Returns      : Pandas dataframe 
    """
    table_name = f"{database}.{schema}.{source_table}"
    # convert table to snowpark dataframe   
    sf_data = session.table(table_name)

    # preprocessing or feature engg

    # Columns with null values and their respective counts
    # null_counts = [
    #     (col_name, sf_data.where(col(col_name).isNull()).count())
    #     for col_name in sf_data.columns
    # ]
    # print(f"Null values in the dataframe: {null_counts}") # no nulls in this dataset, not applying imputer

    # Slice required columns from snowpark dataframe
    sf_data1 = sf_data.select(col("TPEP_DROPOFF_DATETIME").alias("Time_Stamp"), col("TRIP_ID"),
                              col("PASSENGER_COUNT"), col("TRIP_DISTANCE"), col("FARE_AMOUNT"),
                              col("STORE_AND_FWD_FLAG")
                              )

    # cat_cols = ["STORE_AND_FWD_FLAG"]
    
    # OHE = OneHotEncoder(
    #     input_cols=cat_cols,
    #     output_cols=cat_cols,
    #     drop_input_cols=True,
    #     drop="first",
    #     handle_unknown="ignore",
    
    # )
    # sf_data2 = OHE.fit(sf_data1).transform(sf_data1)
    # # rearrage columns
    # sf_data3 = sf_data2[
    #     "TIME_STAMP", "TRIP_ID", "PASSENGER_COUNT", "TRIP_DISTANCE", "FARE_AMOUNT", "STORE_AND_FWD_FLAG_Y"]
    
    # print("Data type: ", type(sf_data3))
    return sf_data1


def create_feature_store(session: snowpark.Session, database: str, name: str,
                         warehouse: str) -> FeatureStore:
    """
    FeatureStore provides APIs to create, materialize, retrieve and manage feature pipelines

    session           : Snowpark session
    database          : Database to create the FeatureStore instance
    name              : schema name i.e Target FeatureStore name (maps to a schema in the database)
                      # In Snowflake, feature stores are represented as schemas. Users can create multiple feature stores as needed
    default_warehouse : Default warehouse for feature store compute
    """

    feature_store = FeatureStore(session, database, name, warehouse, CreationMode.CREATE_IF_NOT_EXIST)
    print(f"feature store: {feature_store} created")
    return feature_store


def delete_entity(feature_store: FeatureStore, name: str, ) -> None:
    """
    Delete a previously registered Entity.
    name : Name of entity to be deleted.

    """
    feature_store.delete_entity(name)
    print(f"Entity: {name} is deleted from {feature_store}")


def create_entity(feature_store: FeatureStore, name: str, join_keys: List[str], desc: str) -> Entity:
    """
    Method creates single Entity instance and register it entity to feature store
    If entity exists in feature store, script generates userwarning i.e UserWarning: Entity TRIP_NUMBER already exists. Skip registration.
    
    feature_store   : FeatureStore to use
    name            : Entity name
    join_keys       : 
    desc            :

    returns         : registered_entity
    """
    entity = Entity(name=name,
                    join_keys=join_keys,
                    desc=desc)
    registered_entity = feature_store.register_entity(entity)
    return registered_entity


def create_entities(feature_store: FeatureStore, entity_parameter_list: List[Dict[str, Any]]) -> Dict[str, Entity]:
    ### if multiple entities to be created, can registe
    """ 
    Entities are the underlying objects that features and feature views are associated with. 
    They encapsulate the join keys used for feature lookups.
        
    """
    entities_mapping = {}
    for entity_parameters in entity_parameter_list:
        entity = create_entity(feature_store=feature_store,
                               name=entity_parameters["name"],
                               join_keys=entity_parameters["join_keys"],
                               desc=entity_parameters["desc"])
        entities_mapping[entity_parameters["name"]] = entity

    return entities_mapping


def delete_feature_view(feature_store: FeatureStore, feature_view: str, version: str) -> None:
    """
    Delete a previously registered FeatureView with name and version.
    feature_view : FeatureView object or name to delete.
    version : Optional version of feature view. Must set when argument feature_view is a str

    """
    print(feature_store.list_feature_views().select('NAME', 'VERSION').show())
    feature_store.delete_feature_view(feature_view, version)
    print(f"FeatureView: {feature_view} is deleted from {feature_store}")
    print(feature_store.list_feature_views().select('NAME', 'VERSION').show())


def create_feature_views(feature_store: FeatureStore, feature_view_parameters: List[Dict[str, Any]],
                         entity_mapping: Dict[str, Entity], feature_df: DataFrame) -> Dict[str, FeatureView]:
    feature_view_mapping = {}
    registered_views = feature_store.list_feature_views()

    for feature_view_param in feature_view_parameters:
        feature_view_name = feature_view_param["name"]
        feature_view_version = feature_view_param["version"]
        entities = [entity_mapping[name] for name in feature_view_param["entities"]]
        feature_df = feature_df
        timestamp_col = feature_view_param.get("timestamp_col")
        refresh_freq = feature_view_param.get("refresh_freq")
        desc = feature_view_param.get("desc")
        feature_desc = feature_view_param.get("feature_desc")

        # If FeatureView already exists in fea_store just return the reference to it
        for view in registered_views:
            if view.name == feature_view_name and view.version == feature_view_version:
                print(f"Feature View : {feature_view_name}_{feature_view_version} already exists")
                break
        else:
            # Create the FeatureView instance
            fv_instance = FeatureView(
                name=feature_view_name,
                entities=entities,
                feature_df=feature_df,
                timestamp_col=timestamp_col,
                refresh_freq=refresh_freq,
                desc=desc).attach_feature_desc(feature_desc)

            # Register the FeatureView instance.  Creates  object in Snowflake
            feature_view = feature_store.register_feature_view(
                feature_view=fv_instance,
                version=feature_view_version,
                block=True,  # whether function call blocks until initial data is available
                overwrite=False,  # whether to replace existing feature view with same name/version
            )

            print(f"Feature View : {feature_view_name}_{feature_view_version} created")
        feature_view_mapping[feature_view_name] = feature_view

    return feature_view_mapping


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    connection_parameters = config["connection_parameters"]
    DATABASE = connection_parameters["database"]
    SCHEMA = connection_parameters["schema"]
    WAREHOUSE = connection_parameters["warehouse"]
    SOURCE_TABLE = config["source_table"]
    FEATURE_STORE_NAME = config["fea_store_name"]

    ENTITY_NAME = config["entity_name"]
    #FEATURE_VIEW_NAME = config["feature_view_name"]
    #VERSION = config["version"]

    ENTITY_JSON_PATH = config["entity_parameters_json"]
    FEATUREVIEW_PARAMS_JSON_PATH = config["featureview_parameters_json"]

    session = snowpark_session(connection_parameters=connection_parameters)
    sf_df = load_data(session, DATABASE, SCHEMA, SOURCE_TABLE)
    print(sf_df.show())

    # sf_df.create_or_replace_dynamic_table(f"{DATABASE}.{SCHEMA}.DIRECT_DYNAMIC_TABLE", warehouse="FEATLK_READ_DEV", lag='12 hours')


    feature_store = create_feature_store(session, DATABASE, FEATURE_STORE_NAME, WAREHOUSE)
    print(feature_store)
    # delete_entity(feature_store, ENTITY_NAME )
    # delete_feature_view(feature_store, FEATURE_VIEW_NAME,VERSION)
    #
    with open(ENTITY_JSON_PATH, "r") as e:
        entity_parameters_list = json.load(e)
    
    entities_dict = create_entities(feature_store, entity_parameters_list)

    with open(FEATUREVIEW_PARAMS_JSON_PATH, "r") as f:
         featureview_parameters = json.load(f)
    
    feature_view_dict = create_feature_views(feature_store=feature_store,
                                              feature_view_parameters=featureview_parameters,
                                              entity_mapping=entities_dict,
                                              feature_df=sf_df
                                              )
    print(feature_view_dict)
