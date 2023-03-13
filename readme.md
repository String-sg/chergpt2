# CherGPT v2

Developed by Joe Tay
Documentation maintained by Kahhow and Adrian
Support and access managed by Lance

```
#This is the secrets.toml file (click on RAW to view the file properly) <br>
#You need to open a MongoDB account and OpenAPI key to deploy this app to streamlit. <br>
#You can develop your own function and call the function in the main() , you need to modify the default tab list in secrets.toml to add in a new function
#--------copy all the info below and put in the secrets settings in streamlit server, remember to get your open API key and MongoDB connection strings

#OpenAI -global numbers

#OpenAI -global numbers

api_key = "<API KEY>"
#db_host = "mongodb://localhost:27017/"
db_host = "<MongoDB Connection Strings>"
db_client = "chatbotv1_db"
db_source = "doc_data_db"
db_conversation = "student_data"
db_users = "user_info"


#chatbot default values
cb_engine = "gpt-3.5-turbo"
cb_temperature = 0.5
cb_max_tokens = 1024
cb_n = 1
cb_presence_penalty = 0.0
cb_frequency_penalty = 0.0
cb_memory = 10

#audio default values
au_engine = "gpt-3.5-turbo"
au_temperature = 0.5
au_max_tokens = 1024
au_n = 1
au_presence_penalty = 0.0
au_frequency_penalty = 0.0
au_duration = 10
au_sample = 44100

#KM default values
km_engine = "gpt-3.5-turbo"
km_temperature = 0.5
km_max_tokens = 2048
km_n = 1
km_presence_penalty = 0.0
km_frequency_penalty = 0.0


#Txt default values
txt_height = 250

engine_options = ["gpt-3.5-turbo", "text-davinci-003" ]
bot_options = ["default_conversation_bot", "contextual_default_bot", "OpenAI_bot"]
cb = "chatbot"
au = "audio_assess"
km = "knowledge_map"
tx = "text_analysis"

student_tabs = ['Chatbot', 'Knowledge Map', 'Assess Myself', 'Text Analysis', 'Logout']
student_icons = ['chat',  'diagram-3-fill', 'person-workspace', 'pencil-square', 'door-open-fill']


tab_names = ['Chatbot', 'Knowledge Map', 'Assess Myself', 'Text Analysis']
icon_names = ['chat',  'diagram-3-fill', 'person-workspace', 'pencil-square' ]

login_names = ['Login']
login_icons = ['door-open']

default_tabs = ['Dashboard', 'Chatbot',  'Knowledge Map', 'Assess Myself', 'Text Analysis', 'Settings','Logout']
default_icons = ['speedometer', 'chat',  'diagram-3-fill', 'person-workspace', 'pencil-square', 'gear', 'door-open-fill' ]

#Values for HayStack Pipelines 
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

#chatbot default settings
cb_bot = "default_conversation_bot"

#Templates for AI LLM with memory 

template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    """

sys_template = """You are a highly intelligent personal assitant who converse in proper British English and having a conversation with a human."""

ast_template = """Pretend you are Jarvis. Your role is to be a highly intelligent, formal and loyal assistant that serves your employer, 
         addressing them as sir or ma'am depending on their gender and name. Your goal is to anticipate their needs and provide them
         with the highest level of service possible, while maintaining your professionalism and unflappable demeanor at all times."""

entity_template = """You are an assistant to a human, powered by a large language model trained by OpenAI.
    You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.
    Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist."""

#URL customisation

URL1 = """https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html"""
URL2 = """https://langchain.readthedocs.io/en/latest/modules/memory/examples/entity_summary_memory.html"""
URL3 = """https://platform.openai.com/docs/guides/chat/introduction""" ```
