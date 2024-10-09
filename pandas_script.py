
import yaml
import json
import pandas as pd
from typing import List, Dict, Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.functions import col
from sklearn.preprocessing import OneHotEncoder

from snowflake.ml.feature_store import FeatureStore, FeatureView, Entity, CreationMode,FeatureViewStatus

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
       print(f"session:",{session})   

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

    # create a DataFrame from data in a table, view, or stream, call the table method 
    sf_data = session.table(table_name)    
    
    # Snowpark DataFrame converted to Pandas DataFrame
    pd_df = sf_data.to_pandas() 

    # do feature transformation
    trip_pd_df = pd_df[["TPEP_DROPOFF_DATETIME","TRIP_ID","PASSENGER_COUNT","TRIP_DISTANCE","FARE_AMOUNT","STORE_AND_FWD_FLAG"]]
    trip_pd_df = trip_pd_df.rename(columns= {"TPEP_DROPOFF_DATETIME":"TIME_STAMP"})

    # # scikit learn operation's tested (OneHotEncoding)    
    # oh= OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    # one_hot_encoded=oh.fit_transform(trip_pd_df[["STORE_AND_FWD_FLAG"]])
    # trans_df = pd.concat([trip_pd_df,one_hot_encoded],axis=1).drop(columns=["STORE_AND_FWD_FLAG"]) 


    # since pandas_df is not accepted in feature_view creation, converting back to snowpark df
    snowpark_df = session.create_dataframe(data=trip_pd_df)
    return snowpark_df

   
    
def create_feature_store(session: snowpark.Session, database: str, name: str,
                         warehouse: str) -> FeatureStore:    
    """
    Method Creates Snowflake Feature Store if not exists and return reference
    session           : Snowpark session
    database          : Database to create the FeatureStore instance
    name              : schema name i.e Target FeatureStore name (maps to a schema in the database)
                      # In Snowflake, feature stores are represented as schemas. Users can create multiple feature stores as needed
    default_warehouse : Default warehouse for feature store compute
    """
    feature_store = FeatureStore(session, database, name, warehouse, CreationMode.CREATE_IF_NOT_EXIST)  
    return feature_store

def cleanup_feature_store(feature_store: FeatureStore):
    """
    Experimental API to delete all entities and feature views in a feature store for easy cleanup
    
    returns: /tmp/snowml/snowflake/ml/feature_store/feature_store.py:190: UserWarning: 
              It will clear ALL feature views and entities in this Feature Store. 
              Make sure your role has sufficient access to all feature views and entities. 
              Insufficient access to some feature views or entities will leave Feature Store in an incomplete state.
                    return f(self, *args, **kargs)

    feature_store    : FeatureStore to delete 
    """
    feature_store._clear(dryrun=False) #  If "dryrun" is set to True (the default) then fs._clear() only prints the objects that will be deleted. 
                                        # If "dryrun" is set to False, it performs the deletion.
    assert feature_store.list_feature_views().count() == 0, "0 feature views left after deletion."
    assert feature_store.list_entities().count() == 0, "0 entities left after deletion."


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
                       
    """ 
    Creates multiple entities

    feature_store          : FeatureStore to use 
    entity_parameter_list  : entities.json 

    """
    entities_mapping = {}
    for entity_parameters in entity_parameter_list:
        entity = create_entity(feature_store=feature_store,
                               name=entity_parameters["name"],
                               join_keys=entity_parameters["join_keys"],
                               desc=entity_parameters["desc"])
        entities_mapping[entity_parameters["name"]] = entity 
      
    return entities_mapping


def delete_entities(feature_store: FeatureStore):
    """
    Note it will check whether there are feature views registered on this entity before it gets deleted, otherwise the deletion will fail.
    """
    for entity in feature_store.list_entities().collect():
        feature_store.delete_entity(entity['NAME'])

    all_entities_df = feature_store.list_entities()
    assert all_entities_df.count() == 0, "0 entities after deletion."
    all_entities_df.show()
    
  
def create_feature_views(feature_store: FeatureStore, feature_view_parameters: List[Dict[str, Any]],
                         entity_mapping: Dict[str, Entity], feature_df: DataFrame) -> Dict[str, FeatureView]:
    
    """
    feature_store            : Name of feature store
    feature_view_parameters  : featureview.json
    entity_mapping           : Dictionary of entity info
    feature_df               : snowpark dataFrame

    """
    print("create_feature_views")
    feature_view_mapping = {}
    registered_views = feature_store.list_feature_views()
    print(len(feature_view_parameters))
    for feature_view_param in feature_view_parameters:
        feature_view_name = feature_view_param["name"]
        feature_view_version = feature_view_param["version"]
        entities = [entity_mapping[name] for name in feature_view_param["entities"]]
        feature_df = feature_df
        timestamp_col = feature_view_param.get("timestamp_col")
        refresh_freq = feature_view_param.get("refresh_freq")
        desc = feature_view_param.get("desc")
        feature_desc = feature_view_param.get("feature_desc")

        print(feature_view_name)
        print(feature_view_version)
        print(entities)
        
    # If FeatureView already exists in fea_store just return the reference to it
        for view in registered_views:
            if view.name == feature_view_name and view.version == feature_view_version:
                print(f"Feature View : {feature_view_name}_{feature_view_version} already exists")
                break
        else:
            print("else")
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
                                feature_view = fv_instance, 
                                version = feature_view_version, 
                                block = True, # whether function call blocks until initial data is available
                                overwrite=False,    # whether to replace existing feature view with same name/version
                        )
            
            print(f"Feature View : {feature_view_name}_{feature_view_version} created")
        feature_view_mapping[feature_view_name] = feature_view

    return feature_view_mapping  

def read_feature_view(feature_store: FeatureStore, feature_view: str, version:str)-> DataFrame:
    """
    reads features data from the registered feature view

    feature_store: Name of feature store
    feature_view: Name of feature view

    returns: snowpark dataframe
    """
    registered_fv = feature_store.get_feature_view(feature_view, version)
    feature_value_df = feature_store.read_feature_view(registered_fv)
    feature_value_df.show()


def suspend_feature_view(feature_store: FeatureStore, feature_view: str, version:str):
    """
    for managed feature views (refresh_freq = "5 minutes"): you can suspend, resume, or manually refresh the backend pipelines. 
    for static feature view (refresh_freq = None) ??
    
    feature_store: Name of feature store
    feature_view: Name of feature view
    verion: version of feature view
    """

    registered_fv = feature_store.get_feature_view(feature_view, version)
    suspended_fv = feature_store.suspend_feature_view(registered_fv)
    assert suspended_fv.status == FeatureViewStatus.SUSPENDED
    print(feature_store.list_feature_views().select('name', 'version', 'desc', 'refresh_freq', 'scheduling_state').show())

def resume_feature_view(feature_store: FeatureStore, suspeneded_feature_view: str):
    """
    for managed feature views (refresh_freq = "5 minutes"): you can suspend, resume, or manually refresh the backend pipelines. 
    for static feature view (refresh_freq = None) ??

    feature_store: Name of feature store
    suspeneded_feature_view: Name of suspended feature view
    
    """
    resumed_fv = feature_store.resume_feature_view(suspeneded_feature_view)
    assert resumed_fv.status == FeatureViewStatus.ACTIVE
    print(feature_store.list_feature_views().select('name', 'version', 'desc', 'refresh_freq', 'scheduling_state').show())


def delete_feature_view(feature_store: FeatureStore, feature_view: str):
    """
    deletes feature view

    feature_store: Name of feature store
    feature_view: Name of feature view
    
    """
    feature_store.delete_feature_view(feature_view)
    print(f"feature_view {feature_view} deleted from {feature_store}")
    print(feature_store.list_feature_views())

def delete_feature_views(feature_store: FeatureStore):
    """
    Warning: Deleting a feature view may break downstream dependencies for other feature views or models that depend on the feature view being deleted.
    deletes multiple feature views

    feature_store: Name of feature store   
    
    """
    
    for view in feature_store.list_feature_views().collect():
        fv = feature_store.get_feature_view(view['NAME'], view['VERSION'])
        feature_store.delete_feature_view(fv)

    all_fvs_df = feature_store.list_feature_views().select('name', 'version') 
    assert all_fvs_df.count() == 0, "0 feature views left after deletion."
    all_fvs_df.show()

###################################################################################################    



if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    connection_parameters = config["connection_parameters"]
    DATABASE = connection_parameters["database"]
    SCHEMA = connection_parameters["schema"]
    WAREHOUSE= connection_parameters["warehouse"]
    SOURCE_TABLE = config["source_table"] 
    FEATURE_STORE_NAME = config["fea_store_name"] 
    ENTITY_JSON_PATH = config["entity_parameters_json"]  
    FEATUREVIEW_PARAMS_JSON_PATH = config["featureview_parameters_json"]

    session= snowpark_session(connection_parameters=connection_parameters)    
    snowpark_df = load_data(session, DATABASE, SCHEMA, SOURCE_TABLE)
    snowpark_df.show()

    feature_store = create_feature_store(session, DATABASE, FEATURE_STORE_NAME, WAREHOUSE)
    print(feature_store)       

    with open(ENTITY_JSON_PATH, "r") as e:
        entity_parameters_list = json.load(e)

    entities_dict = create_entities(feature_store, entity_parameters_list)

    with open(FEATUREVIEW_PARAMS_JSON_PATH, "r") as f:
        featureview_parameters = json.load(f)
    print(featureview_parameters)

    feature_view_dict = create_feature_views(feature_store=feature_store,
                                             feature_view_parameters=featureview_parameters,
                                             entity_mapping=entities_dict,
                                             feature_df=snowpark_df
                                             )
    print(feature_view_dict)



  
