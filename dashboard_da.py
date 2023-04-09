
#Check completed - need to upload pdf to mongodb
import streamlit as st
import pandas as pd
import pymongo
import certifi
from st_aggrid import AgGrid
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]

def dashboard_da():

	st.info("Current student and interaction data (Filter the data and right click to download in CSV or Excel format)")
	try:
		codes = st.session_state.codes_key + [st.session_state.vta_code]
		#st.write(codes)
		documents = list(data_collection.find({"vta_code": {"$in": codes }}, {'_id': 0}))
		df = pd.DataFrame(documents)
		#aggrid_interactive_table(df=df)
		AgGrid(df, height='400px')
	except Exception as e:
		st.write(f"Error: {e}")
		return False
