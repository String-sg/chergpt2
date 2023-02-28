#This is the secrets.toml file
#You need to open a MongoDB account and OpenAPI key to deploy this app to streamlit. 
#You can develop your own function and call the function in the main() , you need to modify the default tab list in secrets.toml to add in a new function
#--------copy all the info below and put in the secrets settings in streamlit server, remember to get your open API key and MongoDB connection strings
#OpenAI -global numbers

api_key = <OPEN API KEY>
#db_host = "mongodb://localhost:27017/" #For testing on local machine, you need to create a .streamlit folder and create a secrets.toml file
db_host = <MONGODB CONNECTION STRINGS FROM MONGODB>
db_client = "chatbotv1_db" #modify your own your_name_db
db_conversation = "student_data"
db_users = "user_info"


#chatbot default values
cb_engine = "text-davinci-003"
cb_temperature = 0.5
cb_max_tokens = 1024
cb_n = 1
cb_presence_penalty = 0
cb_frequency_penalty = 0
cb_memory = 10

#audio default values
au_engine = "text-davinci-003"
au_temperature = 0.5
au_max_tokens = 1024
au_n = 1
au_presence_penalty = 0
au_frequency_penalty = 0
au_duration = 10
au_sample = 44100

#KM default values
km_engine = "text-davinci-003"
km_temperature = 0.5
km_max_tokens = 2048
km_n = 1
km_presence_penalty = 0
km_frequency_penalty = 0


#Txt default values
txt_height = 250


#options

google_api = <GOOGLE API KEY > #Part of LangChain Agent tools (not implemented yet)
goole_cse = <GOOGLE CSE KEY> #Part of Langchain Agent tools (not implemented yet)


engine_options = ["text-davinci-003", "text-davinci-002" ]
cb = "chatbot"
au = "audio_assess"
km = "knowledge_map"
tx = "text_analysis"


tab_names = ['Chatbot', 'Knowledge Map', 'Assess Myself', 'Text Analysis']
icon_names = ['support_agent', 'hub', 'psychology', 'edit_note' ]

default_tabs = ['Login','Dashboard', 'Chatbot',  'Knowledge Map', 'Assess Myself', 'Text Analysis', 'Settings','Logout'] #Can add on new functions
default_icons = ['login','dashboard', 'support_agent',  'hub', 'psychology', 'edit_note', 'settings', 'logout' ] #Need to get icon from Google ICON #https://fonts.google.com/icons

#Values for HayStack Pipelines #not in use for this git resource
#FAISS Store values
hidden_dims = 1536
faiss_index = 'Flat'
faiss_index_path = 'faiss/my_faiss'
faiss_json = 'faiss/my_faiss.json'
haystack_token = 30
haystack_temp = 0.5


#Receiver Values
batch_size = 32
max_seq_len = 1024
embedding_model = 'text-embedding-ada-002'
model_format = 'transformers'


#resource description
data1_path = "data1/Student Handbook.pdf" #modify the resource for data1 
data1_name = "Damai Secondary School Handbook and Information" #modify the search name for data1

data1_description= """This document provides information about Damai Secondary School. It outlines important details about the school's history, 
                    philosophy, vision, mission, motto, values, and strategic thrusts. The document also covers student management and discipline policies and guidelines, 
                    such as school rules and regulations, discipline management, and consequences for misbehavior. The academic program is discussed in detail, 
                    including the STEM-Applied Learning Programme, homework policy, school examination rules and regulations, assessment guidelines, 
                    and academic grading system, among other things. The student development and co-curriculum programme, which includes character and citizenship education, 
                    co-curricular activities, and the Learning for Life Programme, is also covered. Finally, the document provides information about school safety, 
                    including physical education guidelines, road safety, special room safety regulations, and emergency exercises."""

data2_path = "data2/emotions regulation.pdf"
data2_name = "Emotions and feelings regulations and strategies"

data2_description= """This document provides information about coping strategies and emotions regulation."""


data3_path = "data3/class schedule.pdf" #modify the resource for data2
data3_name = "Class Information for tests and homework" #modify the resource for data2

data3_description= """This document is the class scehedule of Mrs Alice Lim of Woonfort Secondary"""

#Templates for AI LLM with memory 

template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    {history}
    Human: {human_input}
    Assistant:"""


template2 = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""
