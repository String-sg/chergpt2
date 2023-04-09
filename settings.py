
import streamlit as st
import hashlib
import pymongo
import certifi
import configparser
import ast
config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
default_bot = config['constants']['cb_bot']
default_engine = config['constants']['cb_engine']
bot_options_str = config.get('menu_lists', 'bot_options')
bot_options = ast.literal_eval(bot_options_str)
engine_options_str = config.get('menu_lists', 'engine_options')
engine_options = ast.literal_eval(engine_options_str)


def general_settings():
	with st.form("General Settings"):
		st.write("**:red[General Settings]**")
		update_api = st.text_input("Please enter your API Key")
		update_password1 = st.text_input("Enter your new password")
		update_password2 = st.text_input("Enter your new password again")
		if update_password1 != update_password2 or update_password1 == "":
			st.error("Password does not match or password blank")
			password_update = False
		else:
			password_update = True
			hashed_password = hashlib.sha256(update_password1.encode()).hexdigest()
		submitted = st.form_submit_button("Save Changes")
		if submitted:
			if password_update == False:
				st.error("Changes not save, please verify your passwords again")
			else:
				st.success("Changes Saved!")
				#user_info_collection = db["user_info"]
				user_info_collection.update_one({"tch_code": st.session_state.vta_code},{"$set": {"pass_key": hashed_password, "api_key": update_api}})


def chatbot_prompt_settings():
	with st.form("Bot Settings Form"):
		st.write("**:green[Bot Settings]**")
		st.warning("""**:red[(Changing these parameters will change the behavior of your Bot)]**""")
		#st.write(st.session_state.bot_key)
		# Dropdown for bot selection

		try:
		    index = bot_options.index(st.session_state.bot_key)
		except ValueError:
		    index = bot_options.index(default_bot)
		bot = st.selectbox('Current Chatbot 🤖', bot_options, index=index)

		try:
		    index = engine_options.index(st.session_state.engine_key)
		except ValueError:
		    index = engine_options.index(default_engine)

		engine = st.selectbox("LLM Engine", options=engine_options, index=index)
		st.write("#")
		submitted = st.form_submit_button("Submit")
		if submitted:
			st.session_state.bot_key = bot
			st.session_state.engine_key = engine
			user_info_collection.update_one({"tch_code": st.session_state.vta_code},{"$set": {"bot_key": bot, "engine_key": engine}})
			st.success("Changes Saved!")


def upload_json_file():
	st.info("Metacognition file template for uploading below")
	uploaded_file = st.file_uploader("Choose a JSON file to upload:", type=["json"])

	if uploaded_file is not None:
		if check_and_upload_json_file(uploaded_file):
			st.success("JSON file uploaded successfully.")
		else:
			st.error("Failed to upload the JSON file.")

def check_and_upload_json_file(uploaded_file):
	directory_path = os.path.join(os.getcwd(), "metacog")

	if os.path.exists(directory_path):
		# Read the uploaded JSON file
		json_data = json.load(uploaded_file)

		# Save the JSON data to the working directory
		file_name = os.path.join(directory_path, "metacog.json")
		with open(file_name, "w") as json_file:
			json.dump(json_data, json_file, ensure_ascii=False, indent=4)
		return True
	else:
		st.warning("The specified directory is not available.")
		return False

@st.cache_resource 
def is_admin_or_metacog_user():
	user_info = user_info_collection.find_one({"tch_code": st.session_state.vta_code})
	
	if user_info and "profile" in user_info:
		if user_info["profile"] in ["administrator", "metacog"]:
			return True
		else:
			return False
	else:
		return False

