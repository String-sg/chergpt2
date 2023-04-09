#CherGpt2
from streamlit_option_menu import option_menu
import streamlit as st
import pymongo
import certifi
import openai

from authenticate import (
	teacher_login, 
	class_login, 
	)
from dashboard import dashboard
from dashboard_da import dashboard_da
from settings import (
	general_settings, 
	chatbot_prompt_settings, 
	upload_json_file,
	is_admin_or_metacog_user
	)

from chatbot import main_bot
from streamlit_extras.colored_header import colored_header
import configparser
import ast

config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
#serper_key = st.secrets["serpapi"]
#serper_key = config['constants']['serpapi']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
st.set_page_config(layout="wide")

c_agent = "chergpt_agent"
ag_agent = "ailc_agent_google"
ab_agent = "ailc_agent_bing"
ar_agent = "ailc_resource_agent"
s_agent = "sourcefinder_agent"
mr_agent = "metacog_resource_agent"
ar_bot = "ailc_resource_bot"
m_bot = "metacog_bot"
mr_bot = "metacog_resource_bot"

def main():
	
	
	if 'tab_key' not in st.session_state:
		st.session_state.tab_key = None
	if 'vta_key' not in st.session_state:
		st.session_state.vta_key = False

	if "temp" not in st.session_state:
		st.session_state["temp"] = ""

	if 'student_key' not in st.session_state:
		st.session_state.student_key = False

	if 'engine_key' not in st.session_state:
		st.session_state.engine_key = config['constants']['cb_engine']

	if 'bot_key' not in st.session_state:
		st.session_state.bot_key = config['constants']['cb_bot']
	
	#config ini values change 
	if 'cb_settings_key' not in st.session_state:
		st.session_state.cb_settings_key = {
			"cb_temperature": float(config['constants']['cb_temperature']),
			"cb_max_tokens": float(config['constants']['cb_max_tokens']),
			"cb_n": float(config['constants']['cb_n']),
			"cb_presence_penalty": float(config['constants']['cb_presence_penalty']),
			"cb_frequency_penalty": float(config['constants']['cb_frequency_penalty'])
		}


	#st.title("✎ Metacognition prototype")
	#st.title("🗣️ GPT3.5/4 Dialogic Agent prototype")
	st.title("✎ CherGpt - Virtual Learning Assistant V2.1)")

	student_tabs_str = config.get('menu_lists', 'student_tabs')
	student_tabs = ast.literal_eval(student_tabs_str)
	student_icons_str = config.get('menu_lists', 'student_icons')
	student_icons = ast.literal_eval(student_icons_str)
	login_names_str = config.get('menu_lists', 'login_names')
	login_names = ast.literal_eval(login_names_str)
	login_icons_str = config.get('menu_lists', 'login_icons')
	login_icons = ast.literal_eval(login_icons_str)
	default_tabs_str = config.get('menu_lists', 'default_tabs')
	default_tabs = ast.literal_eval(default_tabs_str)
	default_icons_str = config.get('menu_lists', 'default_icons')
	default_icons = ast.literal_eval(default_icons_str)

	with st.sidebar: #options for sidebar
		if st.session_state.vta_key != False:
			if st.session_state.student_key != True:
				tab_names = default_tabs 
				icon_names = default_icons 
			else:
				tab_names = student_tabs
				icon_names = student_icons
		else:#not login yet 
			tab_names = login_names#st.secrets["login_names"]
			icon_names = login_icons  #st.secrets["login_icons"]

		tabs  = option_menu(None, tab_names, 
					icons=icon_names, 
					#menu_icon="cast", default_index=0, orientation="vertical",
					styles={
						"container": {"padding": "0!important", "background-color": "#fafafa",  "width": "225px", "margin-left": "0"},
						"icon": {"color": "grey", "font-size": "20px"}, 
						"nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
						"nav-link-selected": {"background-color": "black"},
				}
			)
	if tabs =='Login':
		colored_header(
		label="Students and Teachers Login page",
		description="Enter your student VTA code or teacher ID and password",
		color_name="blue-70",
		)
		placeholder2 = st.empty()
		placeholder3 = st.empty()
		with placeholder2:
			col1, col2 = st.columns([2,2])
			with col1:
				if st.session_state.vta_key == False:
					result = teacher_login()
					if result:
						pass
						placeholder2.empty()
			with col2:
				if st.session_state.vta_key == False:
					if class_login() == True:
						pass
						placeholder2.empty()
		if st.session_state.vta_key == True:

			placeholder3.success("You have logged in successfully!")
			st.success("Please click on the Login word in the sidebar to begin!")

	elif tabs =='Dashboard':
		colored_header(
		label="Data Dashboard ",
		description="Analyse and access your class conversation data",
		color_name="yellow-70",
		)
		if st.session_state.bot_key == ag_agent or st.session_state.bot_key == ab_agent:
			dashboard_da()
		else:
			dashboard()

	elif tabs == 'Chatbot':
		colored_header(
		label="Virtual Learning Assistant 🤖 ",
		description="Conversation agent as a learning support tool",
		color_name="light-blue-70",
		)
		main_bot()
					   

	elif tabs == 'Knowledge Map':
		colored_header(
		label="Knowledge Mapping",
		description="Generating Mindmap using AI",
		color_name="orange-70",
		)

		
		
	elif tabs == 'Assess Myself':
		colored_header(
		label="If you can explain, you can understand",
		description="Using AI to evaluate your explanation",
		color_name="blue-green-70",
		)
		

	elif tabs == 'Settings': #set the different settings for each AI tools {{"settings": "chatbot", "temperature:"},{},{}}
		colored_header(
		label="AI Parameters settings",
		description="Adjust the parameters to fine tune your AI engine",
		color_name="green-70",
		)
		col1, col2 = st.columns([2,2])
		with col1:
			general_settings()
			chatbot_prompt_settings()
			if is_admin_or_metacog_user() == True:
				upload_json_file()
			pass         
		with col2:

			#class_settings()
			#audio_settings()
			pass
		#st.info("Chatbot Settings")
		col1, col2 = st.columns([2,2])
		with col1:
			
			pass
		with col2:
			
			pass
		#st.info("Knowledge Mapping and Assess Myself Settings")
		col1, col2 = st.columns([2,2])
		with col1:
			#au_settings()
			pass
		with col2:
			#km_settings()
			pass            

	elif tabs == 'Logout':
		colored_header(
		label="Logout ",
		description="Clear all settings and exit",
		color_name="red-70",
		)

		st.warning("Click on the logout button to exit the application")
		logout = st.button("Logout")
		if logout:
			for key in st.session_state.keys():
				del st.session_state[key]
			st.experimental_rerun() 


if __name__ == "__main__":
	main()
