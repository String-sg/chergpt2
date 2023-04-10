import streamlit as st
import pymongo
import certifi
import os
from datetime import datetime
import re
import time
import openai
from dialogic_agent import ailc_agent_serp, ailc_agent_bing, ailc_resources_bot, ailc_resource_agent, sourcefinder_agent_serp, metacog_agent, chergpt_agent
from metacog import conversation_starter, conversation_starter_resources
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_extras.stoggle import stoggle
import json
import configparser
import ast
from langchain.vectorstores import Chroma,FAISS

config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
#default conversation table 
CB = config['constants']['CB']


c_agent = "chergpt_agent"
cb_cbot = "chergpt_bot"
ag_agent = "ailc_agent_google"
ab_agent = "ailc_agent_bing"
ar_agent = "ailc_resource_agent"
s_agent = "sourcefinder_agent"
mr_agent = "metacog_resource_agent"
ar_bot = "ailc_resource_bot"
m_bot = "metacog_bot"
mr_bot = "metacog_resource_bot"


#---------------------session states declaration -------------------------

#---------------functions --------------------------------------------

def main_bot():

	openai.api_key  = st.session_state.api_key
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


	if "temp" not in st.session_state:
		st.session_state["temp"] = ""

	if 'document_exist' not in st.session_state:
		st.session_state.document_exist = load_documents()

	if 'data_source' not in st.session_state:
		st.session_state.data_source = None 

	if 'source_bot' not in st.session_state:
		st.session_state.source_bot = False 

	if 's_count' not in st.session_state:
		st.session_state.s_count = 1 

	if 'web_link' not in st.session_state:
		st.session_state.web_link = []

	if 'related_questions' not in st.session_state:
		st.session_state.related_questions = []

	if 'tool_use' not in st.session_state:
		st.session_state.tool_use = False

	if 'doc_tools_names' not in st.session_state:
		st.session_state.doc_tools_names = False

	c1,c2 = st.columns([5,3])
	with c1:
		try:
			st.info("Dialogic Agent")
			chat_history()
			if st.session_state["temp"] != "":
				result = chat_bot(st.session_state["temp"])
				if result != False:
					question, answer = result
					now = datetime.now()
					data_collection.insert_one({"vta_code": st.session_state.vta_code, "function":CB,"question": question, "response": answer, "created_at": now.strftime("%d/%m/%Y %H:%M:%S")})
			#if st.session_state.prompt_first_key == False:
			st.session_state["temp"] = st.text_input("Enter your question", key="text", on_change=clear_text)
		except Exception as e:
			st.error(e)
			return False
	with c2:
		try:
			st.warning("Related Questions & Information")
			show_related_links()
			st.warning("Links and Information")
			show_links()
		except Exception as e:
			st.error(e)
			return False

	
def load_documents():
	try:
		user_info = user_info_collection.find_one({"tch_code": st.session_state.teacher_key})
		if user_info:
			if "db_subject" in user_info and "db_description" in user_info:
				st.session_state.doc_tools_names = {"subject": user_info["db_subject"], "description": user_info["db_description"]}
				st.success('Teacher documents loaded.')
				return True
			else:
				return False
	except Exception as e:
			st.error(e)
			return False

def show_links():
	if not st.session_state.web_link:
		st.write('No links found.')
	else:
		for i in range(0, len(st.session_state.web_link), 2):
			title = st.session_state.web_link[i]
			url = st.session_state.web_link[i+1]
			stoggle(
				f"""<span style="font-weight: normal; color: #187bcd;">{i//2+1}. {title}</span>""",
				f"""<a href="{url}" style="font-weight: normal; color: #187bcd;">{url}</a>""",
			)
			
def show_related_links():
	if not st.session_state.related_questions:
		st.write('No links found.')
	elif st.session_state.source_bot == True:
		for i in range(0, len(st.session_state.related_questions), 2):
			title = st.session_state.related_questions[i]
			content = st.session_state.related_questions[i+1]
			stoggle(
				f"""<span style="font-weight: normal; color: #187bcd;">{i//2+1}. {title}</span>""",
				f"""<span style="font-weight: normal; color: black;">{content}</span>""",
			)
	else:
		for i, question_info in enumerate(st.session_state.related_questions):
			question = question_info.get("question")
			snippet = question_info.get("snippet")
			url = question_info.get("url")
			stoggle(
					f"""<span style="font-weight: normal; color: #187bcd;">{i+1}. {question}</span>""",
					f"""<a href="{url}" style="font-weight: normal; color: #187bcd;">{snippet}</a>""",
				)

def clear_text():
	st.session_state["temp"] = st.session_state["text"]
	st.session_state["text"] = ""

def process_dialougic_agent(text):
	if st.session_state.tool_use == True:
		ref = f'Ref No: ({st.session_state.s_count})'
		st.session_state.s_count += 1
		return f'{text} \n\n {ref}', 'blue'
		st.session_state.tool_use = False
	else:
		st.session_state.s_count += 1
		return f'{text}', 'black'


def process_resource_bot(response):
	answer = response.get('answer', '')
	source_documents = response.get('source_documents', [])

	if source_documents:
		first_doc = source_documents[0]
		source = first_doc.metadata['source']
		topic = first_doc.metadata['topic']
		url = first_doc.metadata['url']
		
		st.session_state.web_link.append(f"Ref No: ({st.session_state.s_count})-{source}: {topic}")
		st.session_state.web_link.append(url)

	if source_documents:
		for document in source_documents:
			source = document.metadata['source']
			topic = document.metadata['topic']
			page = document.metadata['page']
			page_content = document.page_content
			
			st.session_state.related_questions.append(f"Ref No: ({st.session_state.s_count})-{source}: {topic}, Content page {int(page) + 1}")
			st.session_state.related_questions.append(page_content)

		st.session_state.tool_use = True

	if st.session_state.tool_use:
		ref = f'Ref No: ({st.session_state.s_count})'
		st.session_state.s_count += 1
		st.session_state.tool_use = False
		return f'{answer} \n\n {ref}', 'blue'
	else:
		st.session_state.s_count += 1
		return f'{answer}', 'black'




def process_meta_cog(json_data):
	input_text = json_data["input"]
	output = json_data["output"]
	#st.write("Here")
	try:
		data = json.loads(output)
		if 'search_results' in data:
			summaries = [result['summary'] for result in data['search_results']]
			text = ' '.join(summaries)
		else:
			text = output.strip()
	except json.JSONDecodeError:
		text = output.strip()

	if st.session_state.tool_use == True:
		ref = f'Ref No: ({st.session_state.s_count})'
		modified_text = f'(Query: "{input_text}") (Output: {text})'

		#text = follow_up().predict(input=modified_text) #metacog function
		st.session_state.s_count += 1
		return f'{text} \n\n {ref}', 'blue'
	else:
		st.session_state.s_count += 1
		return f'{text}', 'black'

def chat_history():
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []
		"No chat history, you may begin your conversation"

	with st.expander("Click here to see Chat History"):
		messages = st.session_state.chat_msg
		for message in messages:
			#st.write("in msg loop")
			bot_msg = message["response"]
			user_msg = message["question"]
			col_msg = message["colour"]
			st.markdown(f'<div style="text-align: left; color: black; font-weight: bold;"> <span style="color: red;">Chatbot 🤖:</span> <span style="color: {col_msg}; font-weight: normal;">{bot_msg}</span></div>', unsafe_allow_html=True)
			st.markdown(f'<div style="text-align: right;"><span style="color: black; font-weight: normal;">{user_msg}</span><span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)
			st.write("#")


#@st.cache_resource 

def chat_bot(user_input):
	try:
		#st.write(st.session_state.bot_key["cb_bot"])
		if st.session_state.bot_key == c_agent:
			st.session_state.tool_use = False
			ag = chergpt_agent()
			answer = ag.run(input=user_input)
			answer, colour = process_dialougic_agent(answer)
		elif st.session_state.bot_key == cb_cbot:
			st.session_state.tool_use = False
			st.session_state.source_bot = True
			answer = ailc_resources_bot(user_input)
			answer, colour = process_resource_bot(answer)
		elif st.session_state.bot_key == ab_agent:
			st.session_state.tool_use = False
			ag = ailc_agent_bing()
			answer = ag.run(input=user_input)
			answer, colour = process_dialougic_agent(answer)
		elif st.session_state.bot_key == ag_agent:
			st.session_state.tool_use = False
			ag = ailc_agent_serp()
			answer = ag.run(input=user_input)
			answer, colour = process_dialougic_agent(answer)
		elif st.session_state.bot_key == ar_agent:
			st.session_state.tool_use = False
			ag = ailc_resource_agent()
			answer = ag.run(input=user_input)
			answer, colour = process_dialougic_agent(answer)
		elif st.session_state.bot_key == ar_bot:
			st.session_state.tool_use = False
			st.session_state.source_bot = True
			answer = ailc_resources_bot(user_input)
			answer, colour = process_resource_bot(answer)
		elif st.session_state.bot_key == m_bot:
			st.session_state.source_bot = True
			answer = conversation_starter().predict(input=user_input)
			colour = "black"
		elif st.session_state.bot_key == mr_bot:
			st.session_state.tool_use = False
			st.session_state.source_bot = True
			answer = conversation_starter_resources(user_input)
			colour = "black"
		elif st.session_state.bot_key == mr_agent: #to be enhanced with metacog bot
			st.session_state.tool_use = False
			ag = metacog_agent()
			answer = ag.run(input=user_input)
			answer, colour = process_dialougic_agent(answer)
		elif st.session_state.bot_key == s_agent:	
			#activate LLM memeory similar to dialougic agent but no metacog answer , uploading of resources is available 
			st.session_state.tool_use = False
			ag = sourcefinder_agent_serp()
			answer = ag.run(input=user_input)
			answer, colour = process_dialougic_agent(answer)
		if user_input:
			question = user_input
			st.markdown(f'<div style="text-align: left; color: black; font-weight: bold;"> <span style="color: red;">Chatbot 🤖:</span> <span style="color: {colour}; font-weight: normal;">{answer}</span></div>', unsafe_allow_html=True)
			# with placeholder2:
			st.markdown(f'<div style="text-align: right;"><span style="color: black; font-weight: normal;">{question}</span><span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True) 
			#answer = "" 
			
			#data_collection.insert_one({"vta_code": vta_code, "text": question, "response": answer, "error": error, "created_at": dt_string})
			st.session_state.chat_msg.append({ "question": question, "response": answer, "colour": colour})
			return question, answer

	except openai.APIError as e:
		st.error(e)
		return False


	except Exception as e:
		st.error(e)
		return False


#======================= default Q&A source bot =======================================================





