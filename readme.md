#This is the secrets.toml file ( click on RAW to view the file properly )
#You need to open a MongoDB account and OpenAPI key to deploy this app to streamlit. 
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


#promptlayer api keys
promptlayer_api_key = "pl_155c1eebf0b789aae8ea6c1cdd783925"


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


#resource description
data1_path = "data1/Student Handbook.pdf"
data1_name = "Damai Secondary School Handbook and Information"

data1_description= """This document provides information about Damai Secondary School. It outlines important details about the school's history, 
                    philosophy, vision, mission, motto, values, and strategic thrusts. The document also covers student management and discipline policies and guidelines, 
                    such as school rules and regulations, discipline management, and consequences for misbehavior. The academic program is discussed in detail, 
                    including the STEM-Applied Learning Programme, homework policy, school examination rules and regulations, assessment guidelines, 
                    and academic grading system, among other things. The student development and co-curriculum programme, which includes character and citizenship education, 
                    co-curricular activities, and the Learning for Life Programme, is also covered. Finally, the document provides information about school safety, 
                    including physical education guidelines, road safety, special room safety regulations, and emergency exercises."""

data2_path = "data2/nhb_materials.pdf"
data2_name = "National Museum of Singapore exhibition information"

data2_description= """National Museum of Singapore commemorates the 80th anniversary of the Fall of Singapore
with new exhibition relating the untold stories and personal perspectives of the various
people who lived through it. There will be artefacts of the war, documents and exhibits"""


data3_path = "data3/class schedule.pdf"
data3_name = "Class Information for tests and homework"

data3_description= """This document is the class scehedule of Mrs Alice Lim of Woonfort Secondary"""

#chatbot default settings
cb_bot = "default_conversation_bot"

#Templates for AI LLM with memory 

template = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    {history}
    Human: {human_input}
    AI Assistant:
    """

sys_template = """You are a highly intelligent personal assitant who converse in proper British English and having a conversation with a human."""

ast_template = """Pretend you are Jarvis. Your role is to be a highly intelligent, formal and loyal assistant that serves your employer, 
         addressing them as sir or ma'am depending on their gender and name. Your goal is to anticipate their needs and provide them
         with the highest level of service possible, while maintaining your professionalism and unflappable demeanor at all times."""

#ZERO Agent customisation


prefix = """
       Pretend to be a teacher, 
        """

format = """To use a tool, please use the following format:
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        {ai_prefix}: [your response here]"""


suffix = """Begin! Remember to use conversational sentences 
        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

input_variables = ["input", "agent_scratchpad","chat_history"]

ai_prefix = "AI"

human_prefix = "Human"

