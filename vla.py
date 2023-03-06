#version 10 - To include uploadable resource for reference 
from streamlit_option_menu import option_menu
import streamlit as st
import pymongo
import openai
import re
import os
import certifi
import csv
import pandas as pd
from datetime import datetime
import hashlib
import time
import spacy
from plantuml import PlantUML
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mandatory_date_range import date_range_picker
from streamlit_extras.dataframe_explorer import dataframe_explorer
#import hydralit_components as hc
import stylecloud
import spacy_streamlit
from spacy_streamlit import visualize_ner
import whisper
from audio_recorder_streamlit import audio_recorder
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate, VectorDBQA
from langchain.agents import Tool, tool, initialize_agent, ZeroShotAgent, AgentExecutor, ConversationalAgent
from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory, ConversationBufferMemory, ConversationEntityMemory, ConversationSummaryMemory, CombinedMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.document_loaders import PagedPDFSplitter
from langchain.document_loaders import UnstructuredDocxLoader
from langchain.chains import VectorDBQAWithSourcesChain
#import promptlayer
#from promptlayer.langchain.llms import OpenAI


#MongoDB connection Settings
#Connect to MongoDB
db_host = st.secrets["db_host"] #fixed streamlit (v3 - free for user to implement their own DB)
db_client = st.secrets["db_client"] # fixed streamlit ( v3 - free for user to implement their own DB)
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
#default conversation table 
data_collection = db[st.secrets["db_conversation"]] 
user_info_collection = db[st.secrets["db_users"]]

CB = st.secrets["cb"]
AU = st.secrets["au"]
KM = st.secrets["km"]
TX = st.secrets["tx"]

#default settings

#chatbot default values
cb_engine =  st.secrets["cb_engine"]
cb_temperature = st.secrets["cb_temperature"]
cb_max_tokens = st.secrets["cb_max_tokens"]
cb_n = st.secrets["cb_n"]
cb_presence_penalty = st.secrets["cb_presence_penalty"]
cb_frequency_penalty = st.secrets["cb_frequency_penalty"]

#audio default values
au_engine = st.secrets["au_engine"]
au_temperature = st.secrets["au_temperature"]
au_max_tokens = st.secrets["au_max_tokens"]
au_n = st.secrets["au_n"]
au_presence_penalty = st.secrets["au_presence_penalty"]
au_frequency_penalty = st.secrets["au_frequency_penalty"]
au_duration = st.secrets["au_duration"]
au_sample = st.secrets["au_sample"]

#KM default values
km_engine = st.secrets["km_engine"]
km_temperature = st.secrets["km_temperature"]
km_max_tokens = st.secrets["km_max_tokens"]
km_n = st.secrets["km_n"]
km_presence_penalty = st.secrets["km_presence_penalty"]
km_frequency_penalty = st.secrets["km_frequency_penalty"]

#Txt default values
height = st.secrets["txt_height"]

#------------------ Function for assessing myself --------------------------------------------------

def select_language():
    language_options = ["English", "Mandarin", "Malay", "Tamil"]
    default_language = "English"

    # Display the language selection dropdown
    selected_language = st.selectbox("Select your spoken language:", language_options, index=language_options.index(default_language))

    # Map the selected language to a language code
    language_codes = {
        "English": "en",
        "Mandarin": "zh",
        "Malay": "ms",
        "Tamil": "ta"
    }
    selected_language_code = language_codes.get(selected_language, "ms")

    # Display the selected language and language code

    return selected_language, selected_language_code

def record_myself(language):
    sample_rate = st.session_state.au_settings_key["au_sample"]
    my_duration = st.session_state.au_settings_key["au_duration"] #set in settings page
    # Create a button to start recording
    #if st.button("Record"):
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0),
            pause_threshold=my_duration)
    progress_text = ":green[🎙️ Recording in progress...]"
    #my_bar = st.progress(0, text=progress_text)
    if audio_bytes:
        st.write("Press play to hear your recording:")
        st.audio(audio_bytes, format="audio/wav")
        filename = "audio/recording.wav"
        with open(filename, "wb") as f:
            f.write(audio_bytes)

        audio_file= open(filename, "rb")
        result = openai.Audio.transcribe(model="whisper-1", file=audio_file, language=language)

        return result["text"]

def assessment_prompt(prompt, transcript, assessment_type, level, language):
    au_engine, au_temperature, au_max_tokens, au_n, au_presence_penalty, au_frequency_penalty, au_duration, au_sample = st.session_state.au_settings_key.values()
    # Generate GPT response and feedback based on prompt and assessment type
    if assessment_type == "Oral Assessment":
        feedback = f"Provide feedback as a {level} {language} teacher on sentence structure and phrasing to help the student sound more proper."
        #st.write(f"{prompt} " + feedback + "The transcript is :" + transcript)
        response = openai.ChatCompletion.create(
            model=au_engine,
            messages=[{"role": "system", "content": feedback},
                       {"role": "assistant", "content": prompt},
                       {"role": "user", "content": transcript}  ],
            #prompt=f"{prompt} " + feedback + "The transcript is :" + transcript,
            max_tokens=au_max_tokens,
            n=au_n,
            temperature= au_temperature,
            presence_penalty= au_presence_penalty,
            frequency_penalty = au_frequency_penalty)
    elif assessment_type == "Content Assessment":
        feedback = f"Provide feedback as a {level} {language} teacher on the accuracy and completeness of the content explanation, and correct any errors or misconceptions."
        response = openai.ChatCompletion.create(
            model=au_engine,
            #prompt=f"{prompt} " + feedback + "The transcript is :" + transcript,
            messages=[{"role": "system", "content": feedback},
                       {"role": "assistant", "content": prompt},
                       {"role": "user", "content": transcript}  ],
            max_tokens=au_max_tokens,
            n=au_n,
            temperature= au_temperature,
            presence_penalty= au_presence_penalty,
            frequency_penalty = au_frequency_penalty)

    # Display response and feedback to user
    if response is not None:
        #st.write("Assessment prompt: " + prompt)
        answer = response['choices'][0]['message']['content']
        answer = answer.strip()
        #answer = ''.join(answer.splitlines())
        st.write(f"AI Feedback: {answer}")
        return answer
      # st.write("Feedback: " + feedback)


#---------------- Teachers and student log in authentication function ------------------------------------------------------



def teacher_login():
    with st.form(key="authenticate"):
        st.write("Please key in your teacher code and password to access the data")
        teacher_code = st.text_input('Teacher code:')
        teacher_code = teacher_code.lower()
        password = st.text_input('Password:', type="password")
        submit_button = st.form_submit_button(label='Login')
        if submit_button:
            user = user_info_collection.find_one({"tch_code": teacher_code})
            if user:
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                if user["pass_key"] == hashed_password:
                    st.write("Login successful!")
                    vta_codes = [user.get("vta_code{}".format(i), None) for i in range(1, 6)]
                    api_key = user.get("api_key", None)
                    st.session_state.bot_key = user.get("bot_settings", st.session_state.bot_key) 
                    st.session_state.cb_settings_key =  user.get("cb_settings", st.session_state.cb_settings_key )
                    st.session_state.au_settings_key =  user.get("au_settings", st.session_state.au_settings_key )
                    st.session_state.km_settings_key =  user.get("km_settings", st.session_state.km_settings_key )
                    st.session_state.au_settings_key["au_duration"] = user.get("audio_settings",st.secrets["au_duration"])
                    st.session_state.tab_key = st.secrets['default_tabs'], st.secrets['default_icons']
                    st.session_state.login_key = True
                    st.session_state.codes_key = vta_codes
                    st.session_state.api_key = api_key
                    st.session_state.vta_code = teacher_code
                    st.session_state.vta_key = True
                    #st.write(st.session_state.bot_key )
                    return True
                else:
                    st.error("Incorrect password!")
            else:
                st.error("User not found!")
    return False


def class_login():
    with st.form(key='access'):
        st.write("Students, enter your VTA code to access this tool")
        vta_code = st.text_input('VTA code: ')
        vta_code = vta_code.lower()
        submit_button = st.form_submit_button(label='Start')
        if submit_button:
            query = {"$or": [{"vta_code1": vta_code},
                             {"vta_code2": vta_code},
                             {"vta_code3": vta_code},
                             {"vta_code4": vta_code},
                             {"vta_code5": vta_code}]}
            projection = {"_id": 0, "api_key": 1, "name_settings": 1, "icon_settings": 1, "audio_settings": 1, "bot_settings": 1, "default_template": 1, "entity_template": 1, "cb_settings": 1, "au_settings": 1, "km_settings": 1}
            cursor = user_info_collection.find(query, projection)
            for doc in cursor:
                if "api_key" in doc:
                    result = doc["api_key"]
                    tab_names = doc.get("name_settings", st.secrets["student_tabs"])
                    icon_names = doc.get("icon_settings", st.secrets["student_icons"])
                    audio = doc.get("audio_settings")
                    st.session_state.bot_key = doc.get("bot_settings", st.session_state.bot_key)
                    st.session_state.cb_settings_key =  doc.get("cb_settings", st.session_state.cb_settings_key )
                    st.session_state.au_settings_key =  doc.get("au_settings", st.session_state.au_settings_key )
                    st.session_state.km_settings_key =  doc.get("km_settings", st.session_state.km_settings_key )
                    if audio is not None:
                        st.session_state.au_settings_key["au_duration"] = audio
                    st.session_state.tab_key = tab_names, icon_names
                    st.session_state.vta_key = True
                    st.session_state.vta_code = vta_code
                    st.session_state.api_key = result
                    return True
            st.error("VTA code not found, please request for a VTA code before starting your session")
            return False



#@st.cache_resource        
def cache_api(): #assigning openai key  and vta_code from the logins
    openai.api_key  = st.session_state.api_key
    vta_code = st.session_state.vta_code
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # os.environ["PROMPTLAYER_API_KEY"] = st.session_state.promptlayer_api_key
    # promptlayer.api_key  = st.session_state.promptlayer_api_key


#--------------------------------------- Dashboard settings -------------------------------------------------------------------

def generate_wordcloud_from_mongodb(codes): #Generate wordcloud
    #st.subheader("Wordcloud for prompts")
    #

    st.subheader(":blue[WordCloud for all class prompts]")
   
    #code = st.selectbox("Select your class", ["All"] + codes, key="cloud")
    code = "All"

    if code != "Choose a code":
        if code == "All":
            documents = list(data_collection.find({}, {'_id': 0}).sort('_id', pymongo.DESCENDING).limit(100))
        else:
            documents = list(data_collection.find({"vta_code": code}, {"_id": 0}))

        if documents:
            text = " ".join(doc["question"] for doc in documents)
            stylecloud.gen_stylecloud(text,
                              icon_name="fas fa-apple-alt",    
                              colors='black',
                              background_color='white',
                              output_name='images/wordcloud.png',
                              collocations=False,
                              stopwords=True)

            # Display the word cloud using Streamlit
            st.image('images/wordcloud.png')
            

def view_data(codes): #viewing of conversational data 
    #collection = db[collection]
    #st.write("Please choose a code from the list:", codes)
    col1, col2 = st.columns([1,4])
    with col1:
        code = st.selectbox("Select a class or all classes", ["Choose a code"] + codes + ["All"], key="view")
    with col2:
        date_range_picker("Select a date range for your data (demo only)")
    if code != "Choose a code":
        if code == "All":
            documents = list(data_collection.find({"vta_code": {"$in": codes}}, {'_id': 0}))
        else:
            documents = list(data_collection.find({"vta_code": code}, {"_id": 0}))

        if documents:
            df = pd.DataFrame(documents)
            st.info(f"Chatbot and Knowledge Map Data for {code} classes", icon="ℹ️")
            #filtered_df = dataframe_explorer(df)
            #st.dataframe(filtered_df, use_container_width=True)
            st.dataframe(df)
            return code
        else:
            st.info("No documents found", icon="⚠️")

def download_csv(code): #downloading of conversational data

    if code == "All":
        documents = list(data_collection.find({}, {'_id': 0}))
    else:
        documents = list(data_collection.find({"vta_code": code}, {"_id": 0}))

    # Write the documents to a CSV file
    filename = 'all_records.csv'
    with open(filename, "w", newline="") as csvfile:
        fieldnames = documents[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for document in documents:
            writer.writerow(document)

    # Check if the file was written successfully
    try:
        with open(filename, "r") as file:
            data_csv = file.read()
    except:
        st.error("Failed to export records, please try again")
    else:
        st.success("File is ready for downloading")
        st.download_button(
            label="Download data as CSV",
            data=data_csv,
            file_name=filename,
            mime='text/csv',
        )


#--------------------------------------- Chatbot functions ------------------------------------------------------------------


def clear_text():
    st.session_state["temp"] = st.session_state["text"]
    st.session_state["text"] = ""

#@st.cache_resource 
def chat_bot(user_input):
    cb_engine, cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()   
    vta_code = st.session_state.vta_code
    #st.write(st.session_state.bot_key )
    
    with st.expander("Chat History - please close this expander before asking more questions"):
        messages = st.session_state.chat_msg
        for message in messages:
            #st.write("in msg loop")
            bot_msg = message["response"]
            user_msg = message["question"]
            st.write(f":red[Chatbot 🤖]: **{bot_msg}**")
            st.markdown(f'<div style="text-align: right;">{user_msg} <span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)
            st.write("#")
    
    try:

#Tool Settings
        if st.session_state.bot_key["cb_bot"] == "contextual_default_bot":
            response = LLM_entity_response().predict(input=user_input)
        elif st.session_state.bot_key["cb_bot"] == "default_conversation_bot":
            response = LLM_chain_response().predict(human_input=user_input)
        elif st.session_state.bot_key["cb_bot"] == "OpenAI_bot":
            response = openAI_response(user_input)
            response = response['choices'][0]['message']['content']
             
        if user_input:
    
            question = user_input
            answer = response
            #st.write(answer)
            answer = answer.strip()
            #st.write("Question ", question)
            #answer = ''.join(answer.splitlines())
            # with placeholder:
            st.write(f":red[Chatbot 🤖:] **{answer}**")
            # with placeholder2:
            st.markdown(f'<div style="text-align: right;">{question} <span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)  
            error = ""
        
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            #data_collection.insert_one({"vta_code": vta_code, "text": question, "response": answer, "error": error, "created_at": dt_string})
            st.session_state.chat_msg.append({ "question": question, "response": answer})
            return vta_code, question, answer, dt_string

    except openai.APIError as e:
        st.error(e)
        return False


    except Exception as e:
        st.error(e)
        return False


#@st.cache_resource 
def messages(messages):
    for message in reversed(messages):
        #st.write("in msg loop")
        bot_msg = message["response"]
        user_msg = message["text"]
        st.write(f":red[Chatbot 🤖]: **{bot_msg}**")
        st.markdown(f'<div style="text-align: right;">{user_msg} <span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)
        st.write("#")


def openAI_response(user_input):

    response = openai.ChatCompletion.create(
        model=au_engine,
        messages=[{"role": "system", "content": st.session_state.bot_key["cb_system"] },
                   {"role": "assistant", "content": st.session_state.bot_key["cb_assistant"]},
                   {"role": "user", "content": user_input}  ],
        #prompt=f"{prompt} " + feedback + "The transcript is :" + transcript,
        max_tokens=cb_max_tokens,
        n=cb_n,
        temperature= cb_temperature,
        presence_penalty= cb_presence_penalty,
        frequency_penalty = cb_frequency_penalty)

    return response
@st.cache_resource
def default_prompt(_prompt):

    _DEFAULT_TEMPLATE = f"""{_prompt}
                            Current conversation:
                            {{history}}
                            Human: {{human_input}}
                            AI Assistant:"""

    DEFAULT_TEMPLATE = PromptTemplate(
    input_variables=["history", "human_input"],
    template=_DEFAULT_TEMPLATE,
    )

    return DEFAULT_TEMPLATE

@st.cache_resource
def LLM_chain_response():


    llm = OpenAI(
                model_name=cb_engine, 
                temperature=cb_temperature, 
                max_tokens=cb_max_tokens, 
                n=cb_n,
                presence_penalty= cb_presence_penalty,
                frequency_penalty = cb_frequency_penalty
                )


    chatgpt_chain = LLMChain(
                                llm=llm, 
                                prompt=default_prompt(st.session_state.bot_key['cb_template']), 
                                verbose=True, 
                                memory=ConversationalBufferWindowMemory(k=st.secrets["cb_memory"])
                                #memory=memory
                                )

    return chatgpt_chain


@st.cache_resource
def entity_prompt(_prompt):

    _DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = f"""{_prompt}
                                                    Context:
                                                    {{entities}}
                                                    Current conversation:
                                                    {{history}}
                                                    Last line:
                                                    Human: {{input}}
                                                    You:"""

    ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    )

    return ENTITY_MEMORY_CONVERSATION_TEMPLATE





@st.cache_resource
def LLM_entity_response():
    #ENTITY_MEMORY_CONVERSATION_TEMPLATE=st.session_state.bot_key["cb_entity_template"]
    
    
    llm = OpenAI(
                model_name=cb_engine, 
                temperature=cb_temperature, 
                max_tokens=cb_max_tokens, 
                n=cb_n,
                presence_penalty= cb_presence_penalty,
                frequency_penalty = cb_frequency_penalty
                )


    conversation = ConversationChain(
                                    llm=llm, 
                                    verbose=True,
                                    #prompt=str(st.session_state.bot_key["cb_entity_template"]),
                                    prompt = entity_prompt(st.session_state.bot_key['cb_entity']),
                                    #prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                    memory=ConversationEntityMemory(llm=llm)
                                    )

    return conversation





#--------------------------------------- Knowledge Mapping functions -------------------------------------------------------------------



def generate_mindmap():
    km_engine, km_temperature, km_max_tokens, km_n, km_presence_penalty, km_frequency_penalty = st.session_state.km_settings_key.values()
    # Get user input for topic and levels
    topic = st.text_input("Enter a topic to create a knowldege map:")
    levels = st.slider("Enter the number of map levels:", 1, 10, 4)
    #description = st.checkbox("Generate description")
    if st.button('Step 1. Generate knowldege map syntax'):
        if topic == "":
            st.error('Please input a topic')
        else:

            # Create prompt string with user inputs
            #prompt = f"Let's start by creating a simple MindMap on the topic of {topic}. Can you give the mindmap in PlantUML format. Keep it structured from the core central topic branching out to other domains and sub-domains. Let's go to {levels} levels to begin with. Add the start and end mindmap tags and keep it expanding on one side for now."
            prompt = f"""Let's start by creating a simple MindMap on the topic of {topic}. 
            Can you give the mindmap in PlantUML format. Keep it structured from the core central topic branching out to other domains and sub-domains. 
            Let's go to {levels} levels to begin with. Add the start and end mindmap tags and keep it expanding on one side for now. 
            Also, please add color codes to each node based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner. Use the format *[#colour] topic. 
            """

            #     prompt = f"""Analyse this PlantUML syntax: {plantuml}. 
                #                 Can you add a definition to only the leaf nodes? These definitions should be word-wrapped using PlantUML format and not have a surrounding box. 
                #                 Keep the Colors of the nodes as-is, and add additional description nodes to all leaf nodes. The format for adding leaf nodes is ****_ follow by a white space and 'description'
                #                 """
            try:
                # Generate response using OpenAI API
                response = openai.ChatCompletion.create(
                                                model=km_engine, 
                                                messages=[{"role": "user", "content": prompt}],
                                                temperature=km_temperature, 
                                                max_tokens=km_max_tokens, 
                                                n=km_n,
                                                presence_penalty= km_presence_penalty,
                                                frequency_penalty = km_frequency_penalty)

                if response['choices'][0]['message']['content'] != None:
                    msg = response['choices'][0]['message']['content']
                    
                  
                     # Extract PlantUML format string from response
                    plantuml = re.search(r'@startmindmap.*?@endmindmap', msg, re.DOTALL).group()

                # else:
                    #st.write(plantuml)
                return plantuml
            except openai.APIError as e:
                st.error(e)
                st.error("Please type in a new topic or change the words of your topic again")
                return False

            except Exception as e:
                st.error(e)
                st.error("Please type in a new topic or change the words of your topic again")
                return False


# Define a function to render the PlantUML diagram
def render_diagram(uml):
    p = PlantUML("http://www.plantuml.com/plantuml/img/")
    image = p.processes(uml)
    return image


#------------------------------- General Settings Function--------------------------------------------------------------

def select_tabs():
    tab_names = st.secrets["tab_names"]
    icon_names = st.secrets["icon_names"]
    options = dict(zip(tab_names, icon_names))
    #st.write(options)

    # Display the selectbox
    selected_tabs = st.multiselect('Select the tabs that your class will be using under your teacher code', tab_names, default=tab_names)
    if len(selected_tabs) == 0:
        st.error("Please select at least one option")
        return False
    else:

        # Add the logout option
        
        icons = [options[tab] for tab in selected_tabs]
        #st.write(icons)
        selected_tabs.append('Logout')
        icons.append('logout')


        return selected_tabs, icons



st.set_page_config(layout="wide")

def main():

    if "chat_msg" not in st.session_state:
        st.session_state.chat_msg = []

    if "temp" not in st.session_state:
        st.session_state["temp"] = ""

    if "vta_code" not in st.session_state:
        st.session_state.vta_code = None

    if 'vta_key' not in st.session_state:
        st.session_state.vta_key = False

    if 'login_key' not in st.session_state:
        st.session_state.login_key = False

    if 'api_key' not in st.session_state:
        st.session_state.api_key = False

    if 'codes_key' not in st.session_state:
        st.session_state.codes_key = False    

    if 'dashboard_key' not in st.session_state:
        st.session_state.dashboard_key = None  

    if 'temp_uml' not in st.session_state:
        st.session_state.temp_uml = None  

    if 'audio_key' not in st.session_state:
        st.session_state.audio_key = None 

    if 'prompt_bot_key' not in st.session_state:
        st.session_state.prompt_bot_key = None 

    if 'prompt_first_key' not in st.session_state:
        st.session_state.prompt_first_key = True

    if 'tab_key' not in st.session_state:
        st.session_state.tab_key = None


    if 'bot_key' not in st.session_state:
        st.session_state.bot_key = {
                                      "cb_bot": st.secrets["cb_bot"],
                                      "cb_template": st.secrets['template'],
                                      "cb_entity": st.secrets['entity_template'],
                                      "cb_system" : st.secrets['sys_template'],
                                      "cb_assistant" : st.secrets['ast_template']

                                    }

    if 'cb_settings_key' not in st.session_state:
        st.session_state.cb_settings_key = {
                                        "cb_engine": st.secrets["cb_engine"],
                                        "cb_temperature": st.secrets["cb_temperature"],
                                        "cb_max_tokens": st.secrets["cb_max_tokens"],
                                        "cb_n": st.secrets["cb_n"],
                                        "cb_presence_penalty": st.secrets["cb_presence_penalty"],
                                        "cb_frequency_penalty": st.secrets["cb_frequency_penalty"]
                                        } 

    if 'au_settings_key' not in st.session_state:
        st.session_state.au_settings_key = {
                                        "au_engine": st.secrets["au_engine"],
                                        "au_temperature": st.secrets["au_temperature"],
                                        "au_max_tokens": st.secrets["au_max_tokens"],
                                        "au_n": st.secrets["au_n"],
                                        "au_presence_penalty": st.secrets["au_presence_penalty"],
                                        "au_frequency_penalty": st.secrets["au_frequency_penalty"],
                                        "au_duration": st.secrets["au_duration"],
                                        "au_sample": st.secrets["au_sample"]
                                        }

    if 'km_settings_key' not in st.session_state:
        st.session_state.km_settings_key = {
                                        "km_engine": st.secrets["km_engine"],
                                        "km_temperature": st.secrets["km_temperature"],
                                        "km_max_tokens": st.secrets["km_max_tokens"],
                                        "km_n": st.secrets["km_n"],
                                        "km_presence_penalty": st.secrets["km_presence_penalty"],
                                        "km_frequency_penalty": st.secrets["km_frequency_penalty"]
                                        }


    st.title("✎ CherGpt - Virtual Learning Assistant (Beta V2)")
    #st.sidebar.image('images/cotf_logo.png', width=300)
    
    #st.write(":red[Formerly known as CherGPT]")
    #st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
    #st.sidebar.image('images/string.jpeg', width=225)


    with st.sidebar: #options for sidebar
        if st.session_state.tab_key != None:
            tab_names, icon_names = st.session_state.tab_key
            #st.write(st.session_state.tab_key)
        else:
            tab_names = st.secrets["login_names"]
            icon_names = st.secrets["login_icons"]

        tabs  = option_menu(None, tab_names, 
                    icons=icon_names, 
                    menu_icon="cast", default_index=0, orientation="vertical",
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
                if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
                    result = teacher_login()
                    if result:
                        pass
                        placeholder2.empty()
            with col2:
                if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
                    if class_login() == True:
                        cache_api()
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
        # if 'login_key' not in st.session_state or st.session_state.login_key == False:
        #     st.info("Data page accessible by educators only")
        #     st.error("Please log in to access this feature")
        # elif st.session_state.login_key == True:
        col1, col2 = st.columns([1,2])
        with col1:
            generate_wordcloud_from_mongodb(st.session_state.codes_key)

        with col2:
            #choose = st.selectbox("Choose an option", ("Choose", "View Data", "Download Data"))
            code = None
            code = view_data(st.session_state.codes_key)
            if code != None:
                 download_csv(code)
        

    elif tabs == 'Chatbot':
        colored_header(
        label="GPT-3 Chatbot 🤖 ",
        description="OpenAI chabot as a learning support tool",
        color_name="light-blue-70",
        )
        
        prompt = ""
        placeholder3 = st.empty()
        c1,c2 = st.columns([1,2])
        with c1:
            if st.session_state.prompt_first_key == True:
                placeholder4 = st.empty()
                with placeholder4.form("Chatbot Start"):
                    # Ask user for assessment type (Oral or Content)
                    name = st.text_input("Enter your name please")
                    # Ask user for level (Primary 1 to Secondary 5)
                    level = st.selectbox("Choose level:", ["Primary " + str(i) for i in range(1, 7)] + ["Secondary " + str(i) for i in range(1, 6)])
                    # Ask user for subject (English, Mother Tongue, Science, Math, Humanities)
                    subject = st.selectbox("Choose subject:", ["English", "Mother Tongue", "Science", "Math", "Humanities"])
                    # Ask user for topic
                    topic = st.text_input("Based on the subject above, enter a topic that you are learning about:")
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        if topic == "":
                            st.error('Please type in a topic')
                            st.stop()
                        else:
                            st.session_state["temp"] = f"Hi I am {name}, I am a {level} student studying {subject} and would like to know more about {topic}."
                            st.session_state.prompt_bot_key = name, level, subject, topic
                            st.session_state.prompt_first_key = False
                            placeholder4.empty()
        cache_api()
        if st.session_state["temp"] != "":
            #st.write("Inside here")
            result = chat_bot(st.session_state["temp"])
            if result != False:
                vta_code, question, answer, dt_string = result
                name, level, subject, topic = st.session_state.prompt_bot_key
                data_collection.insert_one({"vta_code": vta_code, "function":CB, "topic":topic, "subject":subject, "level":level, "name": name,"question": question, "response": answer, "created_at": dt_string})
            #st.write("**:red[Chat History]**")
            #messages(st.session_state.chat_msg)
        if st.session_state.prompt_first_key == False:
            st.text_input("Enter your question", key="text", value=prompt, on_change=clear_text)


    elif tabs == 'Text Analysis': #complete
        colored_header(
        label="Text Summary and NER detection tool",
        description="NER detection and tokenization",
        color_name="violet-70",
        )
        #models = ["en_core_web_sm", "en_core_web_md"]
        #default_text = "Sundar Pichai is the CEO of Google."
        #spacy_streamlit.visualize(models, default_text)
        # if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
        #     st.info("This tool is available to all teachers but may be restricted to certain students classes")
        #     st.error("Please log in to verify if you are eligible to access this feature")
        # elif st.session_state.vta_key == True:
        text_area = st.text_area("Input your text data", height=height)
        if st.button('Process my text'):
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text_area)
            visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
                    

    elif tabs == 'Knowledge Map':
        colored_header(
        label="Knowledge Mapping",
        description="Generating Mindmap using AI",
        color_name="orange-70",
        )
        
        c1,c2 = st.columns([2,2])
        with c1:
            cache_api()
            st.write("**:red[Step 1. Enter the parameters for your knowldege map]**")
            uml_txt = generate_mindmap()
            st.session_state.temp_uml = uml_txt
        with c2:
            st.write("**:blue[Step 2. Check and confirmed the PlantUML syntax]**")
            uml_input = st.text_area("You may modify the code to before generating the KM", value=st.session_state.temp_uml)
        st.info("Below is the generated Knowledge Map")
        if uml_input != 'None':
            image = render_diagram(uml_input)
            st.image(image)
                    
        
    elif tabs == 'Assess Myself':
        colored_header(
        label="If you can explain, you can understand",
        description="Using AI to evaluate your explanation",
        color_name="blue-green-70",
        )
        
        c1,c2 = st.columns([1,2])
        with c1:
            name = st.text_input("Enter your name:")
            # Ask user for assessment type (Oral or Content)
            assessment_type = st.radio("Choose assessment type:", ("Oral Assessment", "Content Assessment"))
            # Ask user for level (Primary 1 to Secondary 5)
            level = st.selectbox("Choose level:", ["Primary " + str(i) for i in range(1, 7)] + ["Secondary " + str(i) for i in range(1, 6)])
            # Ask user for subject (English, Mother Tongue, Science, Math, Humanities)
            subject = st.selectbox("Choose subject:", ["English", "Mother Tongue", "Science", "Math", "Humanities"])
            # Ask which language
            selected_language, selected_language_code = select_language()
            # Ask user for topic
            topic = st.text_input("Enter topic:")
            #st.session_state.prompt_key = name, assessment_type, level, subject, topic
                        
        with c2:
            uploaded_file = st.file_uploader("Upload an image file to guide your recording")
            if uploaded_file is not None:
                st.image(uploaded_file)

        st.info("Click on the record button to start your recording")
        #name, assessment_type, level, subject, topic = st.session_state.prompt_key
        #st.write(topic)
        if topic == "":
            st.error("Please type in a topic that you would be sharing about")
        else:
            # Generate GPT prompt based on user inputs
            prompt = f"The following transcript is spoken by a {level} student on the subject {subject} for {topic} in {selected_language}."
            #st.write(prompt)
            cache_api()
            transcript = record_myself(selected_language_code)
            
            #st.session_state.prompt_key = prompt, transcript, assessment_type, level
            if transcript == None or transcript == "":
                st.info("Feedback of your explanation (Your feedback may take a while to appear, kindly wait for a response before clicking refresh")
            else:
                if st.button("I am happy with my recordings, please assess me now!"):
                    st.write(f"This is your spoken transcript: :green[{transcript}]")
                    #st.write(transcript)
                    st.info("Feedback of your explanation (Your feedback may take a while to appear, kindly wait for a response before clicking refresh)")
                    answer = assessment_prompt(prompt, transcript, assessment_type, level, selected_language)
                    now = datetime.now()
                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    data_collection.insert_one({"vta_code": st.session_state.vta_code, "function":AU, "topic":topic, "subject":subject, "level":level, "name": name,"question": transcript, "response": answer, "created_at": dt_string})


    elif tabs == 'Settings': #set the different settings for each AI tools {{"settings": "chatbot", "temperature:"},{},{}}
        colored_header(
        label="AI Parameters settings",
        description="Adjust the parameters to fine tune your AI engine",
        color_name="green-70",
        )
       
        col1, col2 = st.columns([2,2])
        with col1:
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
            

                         
        with col2:
            if st.session_state.login_key == True:
                with st.form("Class Settings"):
                    st.write("**:orange[Class Tools Settings]**")
                    
                    result = select_tabs()
                    if result == False:
                        pass
                    else:
                        tabs, icons = result
                                    
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        if result == False:
                            pass
                        else:
                            user_info_collection.update_one({"tch_code": st.session_state.vta_code},{"$set": {"name_settings": tabs, "icon_settings": icons}})
                            st.success("Changes Saved!")
                with st.form("Audio Settings"):
                    st.write("**:blue[Assess Myself Settings]**")
                    value = st.slider('Select the duration of the students recording in secs', 5, 30, st.session_state.au_settings_key["au_duration"], 1)             
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        st.session_state.au_settings_key["au_duration"] = value
                        user_info_collection.update_one({"tch_code": st.session_state.vta_code},{"$set": {"audio_settings": value}})
                        st.success("Changes Saved!")
        st.info("Chatbot Settings")
        col1, col2 = st.columns([2,2])
        with col1:
            with st.form("Bot Settings Form"):
                st.write("**:green[Bot Settings]**")
                st.warning("""**:red[(Changing these parameters will change the behavior of your Bot)]**""")
                #st.write(st.session_state.bot_key)
                bot_settings = st.session_state.bot_key.copy()
                
                # Dropdown for bot selection
                bot_options = st.secrets['bot_options']
                bot_settings["cb_bot"] = st.selectbox('Bot', bot_options, index=bot_options.index(st.session_state.bot_key["cb_bot"]))
                url1 = st.secrets['URL1']
                url2 = st.secrets['URL1']
                url3 = st.secrets['URL3']
                st.markdown("[Information about Conversational Memory Bot](%s)" % url1)
                # Text area for default template
                default_template = st.text_area('Current Template for Conversational Bot', value=bot_settings["cb_template"], height=400)
                revert_default = st.checkbox('Revert to default', key="default_template")
                if revert_default:
                    default_template = st.secrets['template']
                bot_settings["cb_template"] = default_template

                # Text area for entity template
                st.markdown("[Information about Entity Memory Bot](%s)" % url2)
                entity_template = st.text_area('Current Template for Contextual Bot', value=bot_settings["cb_entity"], height=300)
                revert_entity = st.checkbox('Revert to default', key="entity_template")
                if revert_entity:
                    entity_template = st.secrets['entity_template']
                bot_settings["cb_entity"] = entity_template
                
                st.markdown("[Information about OpenAI System](%s)" % url3)
                #Text area for system template
                system_template = st.text_area('OpenAI System Current Template', value=bot_settings["cb_system"], height=150)
                revert_system= st.checkbox('Revert to default', key="system")
                if revert_system:
                    system_template = st.secrets['sys_template']
                bot_settings["cb_system"] = system_template
                st.markdown("[Information about OpenAI Assistant](%s)" % url3)
                #Text area for assistant template
                assistant_template = st.text_area('OpenAI Assistant Current Template', value=bot_settings["cb_assistant"], height=150)
                revert_assist= st.checkbox('Revert to default', key="assistant")
                if revert_assist:
                    assistant_template = st.secrets['ast_template']
                bot_settings["cb_assistant"] = assistant_template

                
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.bot_key = bot_settings
                    user_info_collection.update_one({"tch_code": st.session_state.vta_code}, {"$set": {"bot_settings": bot_settings}})
                    st.success("Changes Saved!")
        with col2:
            with st.form("CB Settings Form"):
                st.write("**:green[Chatbot Parameters Settings]**")
                st.warning("""**:red[(Changing these parameters will change the behavior of your Chatbot)]**""")
                # Dropdown for model
                engine_options = st.secrets['engine_options']
                model = st.selectbox("Model", options=engine_options, index=engine_options.index(st.session_state.cb_settings_key["cb_engine"]))
                # Slider for temperature
                temperature = st.slider('Temperature', min_value=0.1, max_value=5.0, value=st.session_state.cb_settings_key["cb_temperature"], step=0.1)
                # Slider for max tokens
                max_tokens = st.slider('Max Tokens', min_value=10, max_value=2048, value=st.session_state.cb_settings_key["cb_max_tokens"], step=10)
                # Slider for n
                n = st.slider('N', min_value=1, max_value=10, value=st.session_state.cb_settings_key["cb_n"], step=1)
                # Slider for presence penalty
                presence_penalty = st.slider('Presence Penalty', min_value=0.0, max_value=2.0, value=st.session_state.cb_settings_key["cb_presence_penalty"], step=0.1)
                # Slider for frequency penalty
                frequency_penalty = st.slider('Frequency Penalty', min_value=0.0, max_value=2.0, value=st.session_state.cb_settings_key["cb_frequency_penalty"], step=0.1)
                revert = st.checkbox('Warning by checking this box, it will set the default values', key="cb")
                if revert:
                    model = st.secrets["cb_engine"]
                    temperature = st.secrets["cb_temperature"]
                    max_tokens = st.secrets["cb_max_tokens"]
                    n = st.secrets["cb_n"]
                    presence_penalty = st.secrets["cb_presence_penalty"]
                    frequency_penalty = st.secrets["cb_frequency_penalty"]

                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.cb_settings_key = {
                        "cb_engine": model,
                        "cb_temperature": temperature,
                        "cb_max_tokens": max_tokens,
                        "cb_n": n,
                        "cb_presence_penalty": presence_penalty,
                        "cb_frequency_penalty": frequency_penalty
                    }
                    user_info_collection.update_one({"tch_code": st.session_state.vta_code}, {"$set": {"cb_settings": st.session_state.cb_settings_key}})
                    st.success("Changes Saved!")


        st.info("Knowledge Mapping and Assess Myself Settings")
        col1, col2 = st.columns([2,2])
        with col1:
            with st.form("KM Settings Form"):
                st.write("**:green[Knowledge Mapping Parameters Settings]**")
                st.warning("""**:red[(Changing these parameters will change the behavior of your Knowledge Model)]**""")
                st.write("Model options:")
                model_options = st.secrets['engine_options']
                km_engine = st.selectbox('Model', model_options, key="km_engine")
                km_temperature = st.slider('Temperature', min_value=0.1, max_value=1.0, step=0.1, value=st.session_state.km_settings_key["km_temperature"])
                km_max_tokens = st.slider('Max Tokens', min_value=10, max_value=2048, step=10, value=st.session_state.km_settings_key["km_max_tokens"])
                km_n = st.slider('N-gram Size', min_value=1, max_value=5, step=1, value=st.session_state.km_settings_key["km_n"])
                km_presence_penalty = st.slider('Presence Penalty', min_value=0.0, max_value=2.0, step=0.1, value=st.session_state.km_settings_key["km_presence_penalty"])
                km_frequency_penalty = st.slider('Frequency Penalty', min_value=0.0, max_value=2.0, step=0.1, value=st.session_state.km_settings_key["km_frequency_penalty"])
                revert = st.checkbox('Warning by checking this box, it will set the default values', key="km")
                if revert:
                    model = st.secrets["km_engine"]
                    temperature = st.secrets["km_temperature"]
                    max_tokens = st.secrets["km_max_tokens"]
                    n = st.secrets["km_n"]
                    presence_penalty = st.secrets["km_presence_penalty"]
                    frequency_penalty = st.secrets["km_frequency_penalty"]
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.km_settings_key = {
                        "km_engine": km_engine,
                        "km_temperature": km_temperature,
                        "km_max_tokens": km_max_tokens,
                        "km_n": km_n,
                        "km_presence_penalty": km_presence_penalty,
                        "km_frequency_penalty": km_frequency_penalty,
                    }
                    user_info_collection.update_one({"tch_code": st.session_state.vta_code}, {"$set": {"km_settings": st.session_state.km_settings_key}})
                    st.success("Changes Saved!")
        with col2:
            with st.form("AU Settings Form"):
                st.write("**:green[Assess Myself Parameters Settings]**")
                st.warning("""**:red[(Changing these parameters will change the behavior of your AU model)]**""")
                st.write("Model options:")
                model_options = st.secrets['engine_options']
                au_engine = st.selectbox('Model', model_options, key="au_engine")
                au_temperature = st.slider('Temperature', min_value=0.1, max_value=1.0, step=0.1, value=st.session_state.au_settings_key["au_temperature"])
                au_max_tokens = st.slider('Max Tokens', min_value=10, max_value=2048, step=10, value=st.session_state.au_settings_key["au_max_tokens"])
                au_n = st.slider('N-gram Size', min_value=1, max_value=5, step=1, value=st.session_state.au_settings_key["au_n"])
                au_presence_penalty = st.slider('Presence Penalty', min_value=0.0, max_value=2.0, step=0.1, value=st.session_state.au_settings_key["au_presence_penalty"])
                au_frequency_penalty = st.slider('Frequency Penalty', min_value=0.0, max_value=2.0, step=0.1, value=st.session_state.au_settings_key["au_frequency_penalty"])
                revert = st.checkbox('Warning by checking this box, it will set the default values', key="au")
                if revert:
                    model = st.secrets["au_engine"]
                    temperature = st.secrets["au_temperature"]
                    max_tokens = st.secrets["au_max_tokens"]
                    n = st.secrets["au_n"]
                    presence_penalty = st.secrets["au_presence_penalty"]
                    frequency_penalty = st.secrets["au_frequency_penalty"]
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.session_state.au_settings_key = {
                        "au_engine": au_engine,
                        "au_temperature": au_temperature,
                        "au_max_tokens": au_max_tokens,
                        "au_n": au_n,
                        "au_presence_penalty": au_presence_penalty,
                        "au_frequency_penalty": au_frequency_penalty,
                    }
                    user_info_collection.update_one({"tch_code": st.session_state.vta_code}, {"$set": {"au_settings": st.session_state.au_settings_key}})
                    st.success("Changes Saved!")


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
