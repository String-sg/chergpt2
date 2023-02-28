#version 7 - Customise ChatBot with memory
from st_on_hover_tabs import on_hover_tabs
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
from langchain.agents import Tool, tool
from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory, ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.document_loaders import PagedPDFSplitter
from langchain.document_loaders import UnstructuredDocxLoader
from langchain.chains import VectorDBQAWithSourcesChain


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

def record_myself():
    sample_rate = st.session_state.au_settings_key["au_sample"]
    if st.session_state.audio_key != None:
        my_duration = st.session_state.audio_key
    else:
        my_duration = st.session_state.au_settings_key["au_duration"]
    # Create a button to start recording
    #if st.button("Record"):
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0),
            pause_threshold=my_duration)
    progress_text = ":green[🎙️ Recording in progress...]"
    #my_bar = st.progress(0, text=progress_text)
    if audio_bytes:
        st.write("Playback:")
        st.audio(audio_bytes, format="audio/wav")
        filename = "audio/recording.wav"
        with open(filename, "wb") as f:
            f.write(audio_bytes)


        model = whisper.load_model("base")
        result = model.transcribe(filename)

        return result["text"]

def assessment_prompt(prompt, transcript, assessment_type, level):
    au_engine, au_temperature, au_max_tokens, au_n, au_presence_penalty, au_frequency_penalty, au_duration, au_sample = st.session_state.au_settings_key.values()
    # Generate GPT response and feedback based on prompt and assessment type
    if assessment_type == "Oral Assessment":
        feedback = f"Provide feedback as a {level} teacher on sentence structure and phrasing to help the student sound more proper."
        #st.write(f"{prompt} " + feedback + "The transcript is :" + transcript)
        response = openai.Completion.create(
            engine=au_engine,
            prompt=f"{prompt} " + feedback + "The transcript is :" + transcript,
            max_tokens=au_max_tokens,
            n=au_n,
            temperature= au_temperature,
            presence_penalty= au_presence_penalty,
            frequency_penalty = au_frequency_penalty)
    elif assessment_type == "Content Assessment":
        feedback = f"Provide feedback as a {level} teacher on the accuracy and completeness of the content explanation, and correct any errors or misconceptions."
        response = openai.Completion.create(
            engine=au_engine,
            prompt=f"{prompt} " + feedback + "The transcript is :" + transcript,
            max_tokens=au_max_tokens,
            n=au_n,
            temperature= au_temperature,
            presence_penalty= au_presence_penalty,
            frequency_penalty = au_frequency_penalty)

    # Display response and feedback to user
    if response is not None:
        #st.write("Assessment prompt: " + prompt)
        answer = response["choices"][0]["text"]
        answer = answer.strip()
        #answer = ''.join(answer.splitlines())
        st.write(f"AI Feedback: :violet[{answer}]")
        return answer
      # st.write("Feedback: " + feedback)


#---------------- Teachers log in authentication function ------------------------------------------------------

def login(db, username, password):
    # user_info_collection = db["user_info"]
    user = user_info_collection.find_one({"tch_code": username})
    if user:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if user["pass_key"] == hashed_password:
            st.write("Login successful!")
            vta_codes = [user.get("vta_code{}".format(i), None) for i in range(1, 6)]
            api_key = user.get("api_key", None)
            return vta_codes, api_key
        else:
            st.write("Incorrect password!")
            return None, None
    else:
        st.write("User not found!")
        return None, None


def teacher_download():
   
    with st.form(key="authenticate"):
        st.write("Please key in your teacher code and password to access the data")
        teacher_code = st.text_input('Teacher code:')
        password = st.text_input('Password:', type="password")
        submit_button = st.form_submit_button(label='Login')
        if submit_button:
            vta_codes, api_key = login(db, teacher_code, password)
            if vta_codes and api_key:
                st.session_state.login_key = True
                return vta_codes, api_key, teacher_code
            else:
                st.error("Incorrect teacher code or password")

#------------------------------------------- Student VTA Code Log in  ------------------------------------------------------

# def get_api_key_by_vta_code(vta_code, collection):
#     query = {"$or": [{"vta_code1": vta_code},
#                      {"vta_code2": vta_code},
#                      {"vta_code3": vta_code},
#                      {"vta_code4": vta_code},
#                      {"vta_code5": vta_code}]}
#     projection = {"_id": 0, "api_key": 1}
#     cursor = collection.find(query, projection)
#     for doc in cursor:
#         if "api_key" in doc:
#             return doc["api_key"]
#     return False


def get_api_key_by_vta_code(vta_code, collection):
    query = {"$or": [{"vta_code1": vta_code},
                     {"vta_code2": vta_code},
                     {"vta_code3": vta_code},
                     {"vta_code4": vta_code},
                     {"vta_code5": vta_code}]}
    projection = {"_id": 0, "api_key": 1, "name_settings": 1, "icon_settings": 1, "audio_settings": 1}
    cursor = collection.find(query, projection)
    for doc in cursor:
        if "api_key" in doc:
            return doc["api_key"], doc.get("name_settings"), doc.get("icon_settings"), doc.get("audio_settings")
    return False, None, None, None



#checking if vta_code exist
def check_vta_code():
    with st.form(key='access'):
        #st.write("Welcome to GPT-3 chatbot ")
        st.write("Students, enter your VTA code to access this tool")
        st.write("Teachers, log in to the dashboard to access this tool")
        vta_code = st.text_input('VTA code: ')
        vta_code = vta_code.lower()
        submit_button = st.form_submit_button(label='Start')

        if submit_button:

            #cursor.execute('''SELECT * FROM users_db WHERE vta_code = ?''', (vta_code,))
            #result = cursor.fetchone()
            # user_info_collection = db["user_info"] #need to change and set it in secret
            result = get_api_key_by_vta_code(vta_code, user_info_collection)
            api, tab_names, icon_names, audio = result
            #result = None
            #vta_code = "joe"
            #result = "found"
            # Check if the vta_code exists in the sheet_db table
            if api is False:
                st.error("vta_code not found, please request for a VTA code before starting your session")
                #st.session_state.vta_key == False
            else:
                #st.success("vta_code found in the sheet_db table: ")
                #st.write(result)
                # put into session tab code if tab_names are not None
                if tab_names != None and icon_names != None:
                    st.session_state.tab_key = tab_names, icon_names
                if audio != None:
                    st.session_state.audio_key = audio
                st.session_state.vta_key = True
                st.session_state.vta_code = vta_code
                st.session_state.api_key = api
                return True

#@st.cache_resource        
def cache_api(): #assigning openai key  and vta_code from the logins
    openai.api_key  = st.session_state.api_key
    vta_code = st.session_state.vta_code
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
            documents = list(data_collection.find({}, {'_id': 0}))
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


@tool
def search_doc(query: str) -> str: #further exploration need - referencing document
    """
    Search a machine learning model with the given query and return the answer with sources as a string.
    """
    data2 = load_chain(st.secrets["data2_path"], st.secrets["data2_name"])
    my_dict = data2({"question": query}, return_only_outputs=True)
    # answer = my_dict['answer'].strip()
    # sources = my_dict['sources']
    # result = f"{answer} (sources: {sources})"
    return my_dict


#@st.cache_resource 
def chat_bot(user_input):
    cb_engine, cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()   
    data1 = load_FAISS_store(st.secrets["data1_path"], st.secrets["data1_name"])    
    data2 = load_FAISS_store(st.secrets["data2_path"], st.secrets["data2_name"])


    tools = [ 
        Tool(
        name = st.secrets["data1_name"],
        func=data1.run,
        description=st.secrets['data1_description'],
        #return_direct=True
        ),
        Tool(
        name = st.secrets["data2_name"],
        func=data2.run,
        description=st.secrets['data2_description'],
        #return_direct=True,
        ),

    ]


    vta_code = st.session_state.vta_code
    
    with st.expander("Chat History - please close this expander before asking more questions"):
        messages = st.session_state.chat_msg
        for message in messages:
            #st.write("in msg loop")
            bot_msg = message["response"]
            user_msg = message["question"]
            st.write(f":red[Chatbot 🤖]: **{bot_msg}**")
            st.markdown(f'<div style="text-align: right;">{user_msg} <span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)
            st.write("#")
    
    #response = extractive_chatbot(user_input)
    # response = openai.Completion.create(
    #                                     engine=cb_engine, 
    #                                     prompt=f"Pretend you are a {level} teacher, " + user_input, 
    #                                     temperature=cb_temperature, 
    #                                     max_tokens=cb_max_tokens, 
    #                                     n=cb_n,
    #                                     presence_penalty= cb_presence_penalty,
    #                                     frequency_penalty = cb_frequency_penalty)
    agent_chain = agent_response(tools)
    response = agent_chain.run(input=user_input)
    #response = agent_chain({"query": user_input})
                                        
    if user_input:
        try:

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
            # question = user_input
            # answer = ""
            # error = f"ApiError: {e}"
            # now = datetime.now()
            # # dd/mm/YY H:M:S
            # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # #data_collection.insert_one({"vta_code": vta_code, "text": question, "response": answer, "error": error, "created_at": dt_string})
            return False


        except Exception as e:
            st.error(e)
            # question = user_input
            # answer = ""
            # error = f"Unexpected Error: {e}"
            # now = datetime.now()
            # # dd/mm/YY H:M:S
            # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # #data_collection.insert_one({"vta_code": vta_code, "text": question, "response": answer, "error": error, "created_at": dt_string})
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


# @st.cache_resource
# def LLM_chain_response():
#     prompt = PromptTemplate(
#                             input_variables=["history", "human_input"], 
#                             template=st.secrets["template"]
#                             )

#     llm = OpenAI(
#                 model_name=cb_engine, 
#                 temperature=cb_temperature, 
#                 max_tokens=cb_max_tokens, 
#                 n=cb_n,
#                 presence_penalty= cb_presence_penalty,
#                 frequency_penalty = cb_frequency_penalty
#                 )


#     chatgpt_chain = LLMChain(
#                                 llm=llm, 
#                                 prompt=prompt, 
#                                 verbose=True, 
#                                 memory=ConversationalBufferWindowMemory(k=st.secrets["cb_memory"])
#                                 #memory=memory
#                                 )

#     return chatgpt_chain


@st.cache_resource  
def load_FAISS_store(_path, _name): #demo1
    llm = OpenAI(
                model_name=cb_engine, 
                temperature=0, 
                max_tokens=cb_max_tokens, 
                n=cb_n,
                presence_penalty= cb_presence_penalty,
                frequency_penalty = cb_frequency_penalty
                )
    
    loader = PagedPDFSplitter(_path)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings, collection_name=_name)
    d_search = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch)

    return d_search

@st.cache_resource  
def load_chain(_path, _name): #demo1
    llm = OpenAI(
                model_name=cb_engine, 
                temperature=0, 
                max_tokens=cb_max_tokens, 
                n=cb_n,
                presence_penalty= cb_presence_penalty,
                frequency_penalty = cb_frequency_penalty
                )
    
    loader = PagedPDFSplitter(_path)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings, metadata=[{"source": f"{i}-pl"} for i in range(len(texts))], collection_name=_name)
    chain = VectorDBQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)

    return chain






@st.cache_resource
def agent_response(_tools):

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm = OpenAI(
                model_name=cb_engine, 
                temperature=0, 
                max_tokens=cb_max_tokens, 
                n=cb_n,
                presence_penalty= cb_presence_penalty,
                frequency_penalty = cb_frequency_penalty
                )


    agent_chain = initialize_agent(tools=_tools,
                                llm=llm, 
                                #prompt=prompt,
                                agent="conversational-react-description",
                                verbose=True, 
                                memory=memory)
                                #memory=memory
                                

    return agent_chain



#--------------------------------------- Knowledge Mapping functions -------------------------------------------------------------------



def generate_mindmap():
    km_engine, km_temperature, km_max_tokens, km_n, km_presence_penalty, km_frequency_penalty = st.session_state.km_settings_key.values()
    # Get user input for topic and levels
    topic = st.text_input("Enter a topic to create a knowldege map:")
    levels = st.slider("Enter the number of map levels:", 1, 10, 4)
    #description = st.checkbox("Generate description")
    if st.button('Generate syntax and knowldege map'):

        # Create prompt string with user inputs
        #prompt = f"Let's start by creating a simple MindMap on the topic of {topic}. Can you give the mindmap in PlantUML format. Keep it structured from the core central topic branching out to other domains and sub-domains. Let's go to {levels} levels to begin with. Add the start and end mindmap tags and keep it expanding on one side for now."
        prompt = f"""Let's start by creating a simple MindMap on the topic of {topic}. 
        Can you give the mindmap in PlantUML format. Keep it structured from the core central topic branching out to other domains and sub-domains. 
        Let's go to {levels} levels to begin with. Add the start and end mindmap tags and keep it expanding on one side for now. 
        Also, please add color codes to each node based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner. Use the format *[#colour] topic. 
        """

        # Generate response using OpenAI API
        response = openai.Completion.create(
                                        engine=km_engine, 
                                        prompt=prompt, 
                                        temperature=km_temperature, 
                                        max_tokens=km_max_tokens, 
                                        n=km_n,
                                        presence_penalty= km_presence_penalty,
                                        frequency_penalty = km_frequency_penalty)

        if response.choices[0].text != None:
            #st.write(response.choices[0].text)

            # Extract PlantUML format string from response
             # Extract PlantUML format string from response
            plantuml = re.search(r'@startmindmap.*?@endmindmap', response.choices[0].text, re.DOTALL).group()

            # Display PlantUML code in Streamlit app

            # if description == True:
            #     prompt = f"""Analyse this PlantUML syntax: {plantuml}. 
            #                 Can you add a definition to only the leaf nodes? These definitions should be word-wrapped using PlantUML format and not have a surrounding box. 
            #                 Keep the Colors of the nodes as-is, and add additional description nodes to all leaf nodes. The format for adding leaf nodes is ****_ follow by a white space and 'description'
            #                 """

            #     # Generate response using OpenAI API
            #     response = openai.Completion.create(
            #         engine=cb_engine,
            #         prompt=prompt,
            #         temperature=0.5,
            #         max_tokens=2048,
            #         top_p=1,
            #         frequency_penalty=0,
            #         presence_penalty=0
            #     )


            #     plantuml = re.search(r'@startmindmap.*?@endmindmap', response.choices[0].text, re.DOTALL).group()
            #     return plantuml

            # else:
            return plantuml


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


    st.title("✎ CherGPT V2 - Virtual Learning Assistant")
    #st.sidebar.image('images/cotf_logo.png', width=300)
    
    #st.write(":red[Formerly known as CherGPT]")
    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
    st.sidebar.image('images/string.jpeg', width=50)


    with st.sidebar: #options for sidebar
        if st.session_state.tab_key != None:
            tab_names, icon_names = st.session_state.tab_key
            #st.write(st.session_state.tab_key)
        else:
            tab_names = st.secrets["default_tabs"]
            icon_names = st.secrets["default_icons"]

        tabs = on_hover_tabs(tabName=tab_names, # Dashboard will have chat access and data from the questions, midmap, Media Analysis, Assess Myself
                             iconName=icon_names, default_choice=0)

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
                    result = teacher_download()
                    if result:
                        codes, api, tch_code = result
                        st.session_state.api_key = api
                        st.session_state.codes_key = codes
                        st.session_state.vta_code = tch_code
                        st.session_state.vta_key = True
                        placeholder2.empty()
            with col2:
                if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
                    if check_vta_code() == True:
                        cache_api()
                        placeholder2.empty()
        if st.session_state.vta_key == True:
            placeholder3.success("You have logged in successfully")


    elif tabs =='Dashboard':
        colored_header(
        label="Data Dashboard ",
        description="Analyse and access your class conversation data",
        color_name="yellow-70",
        )
        if 'login_key' not in st.session_state or st.session_state.login_key == False:
            st.info("Data page accessible by educators only")
            st.error("Please log in to access this feature")
        elif st.session_state.login_key == True:
            col1, col2 = st.columns([1,2])
            with col1:
                generate_wordcloud_from_mongodb(st.session_state.codes_key)

            with col2:
                #choose = st.selectbox("Choose an option", ("Choose", "View Data", "Download Data"))
                code = None
                code = view_data(st.session_state.codes_key)
                if code != None:
                     download_csv(code)


            # #can apply customisation to almost all the properties of the card, including the progress bar
            # theme_bad = {'bgcolor': '#FFF0F0','title_color': 'red','content_color': 'red','icon_color': 'red', 'icon': 'fa fa-times-circle'}
            # theme_neutral = {'bgcolor': '#f9f9f9','title_color': 'orange','content_color': 'orange','icon_color': 'orange', 'icon': 'fa fa-question-circle'}
            # theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}

            
            #st.info("Demo only = feature is being worked on to analyse the sentiment of the conversations of the students")
            #cc = st.columns(3)

            # with cc[0]:
            #  # can just use 'good', 'bad', 'neutral' sentiment to auto color the card
            #  hc.info_card(title='POSITIVE & GOOD', content='All good!', sentiment='good',bar_value=77)

            # with cc[1]:
            #  hc.info_card(title='NEUTRAL & OBJECTIVE', content='Normal', sentiment='neutral',bar_value=55)

            # with cc[2]:
            #  hc.info_card(title='NEGATIVE & POOR', content='Negative feelings',bar_value=12,theme_override=theme_bad)

        

    elif tabs == 'Chatbot':
        colored_header(
        label="GPT-3 Chatbot 🤖 ",
        description="OpenAI chabot as a learning support tool",
        color_name="light-blue-70",
        )

        


        #if teacher has login skip this part
        if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
            st.info("This tool is available to all teachers but may be restricted to certain students classes")
            st.error("Please log in to verify if you are eligible to access this feature")
        elif st.session_state.vta_key == True:
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
        if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
            st.info("This tool is available to all teachers but may be restricted to certain students classes")
            st.error("Please log in to verify if you are eligible to access this feature")
        elif st.session_state.vta_key == True:
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
        if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
            st.info("This tool is available to all teachers but may be restricted to certain students classes")
            st.error("Please log in to verify if you are eligible to access this feature")
        elif st.session_state.vta_key == True:
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
        label="Feynman Principle of explanation",
        description="Using AI to evaluate your explanation",
        color_name="blue-green-70",
        )
        pass
        if 'vta_key' not in st.session_state or st.session_state.vta_key != True:
            st.info("This tool is available to all teachers but may be restricted to certain students classes")
            st.error("Please log in to verify if you are eligible to access this feature")
        elif st.session_state.vta_key == True:
            c1,c2 = st.columns([1,2])
            with c1:
                name = st.text_input("Enter your name:")
                # Ask user for assessment type (Oral or Content)
                assessment_type = st.radio("Choose assessment type:", ("Oral Assessment", "Content Assessment"))
                # Ask user for level (Primary 1 to Secondary 5)
                level = st.selectbox("Choose level:", ["Primary " + str(i) for i in range(1, 7)] + ["Secondary " + str(i) for i in range(1, 6)])
                # Ask user for subject (English, Mother Tongue, Science, Math, Humanities)
                subject = st.selectbox("Choose subject:", ["English", "Mother Tongue", "Science", "Math", "Humanities"])

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
                prompt = f"The following transcript is spoken by a {level} student on the subject {subject} for {topic}."
                #st.write(prompt)
                cache_api()
                transcript = record_myself()
                #st.session_state.prompt_key = prompt, transcript, assessment_type, level
                if transcript == None or transcript == "":
                    st.info("Feedback of your explanation (Your feedback may take a while to appear, kindly wait for a response before clicking refresh")
                else:
                    st.write(f"This is your spoken transcript: :green[{transcript}]")
                    #st.write(transcript)
                    st.info("Feedback of your explanation (Your feedback may take a while to appear, kindly wait for a response before clicking refresh)")
                    answer = assessment_prompt(prompt, transcript, assessment_type, level)
                    now = datetime.now()
                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    data_collection.insert_one({"vta_code": st.session_state.vta_code, "function":AU, "topic":topic, "subject":subject, "level":level, "name": name,"question": transcript, "response": answer, "created_at": dt_string})
                #assessment_prompt(prompt, transcript, assessment_type, level)

    # elif tabs == 'Data Analytics':
    #     colored_header(
    #     label="Data analyser",
    #     description="Data tools to auto generate trends and patterns visually",
    #     color_name="yellow-70",
    #     )
    #     #st.write("**This tool will auto generate statistics on your data in CSV format**")
    #     st.info("Note that your data will not be stored in the server, click x to remove the file")
    #     st.warning(":red[ ⚠️ Do not upload any confidential or sensitive data]")
    #     c1,c2 = st.columns([2,2])

    #     with c1:
    #         if 'vta_key' not in st.session_state or st.session_state.vta_key == False:
    #             placeholder3 = st.empty()
    #             with placeholder3:
    #                 if check_vta_code() == True:
    #                     placeholder3.empty()
    #         if st.session_state.vta_key == True:

    #             file = st.file_uploader("Select a CSV file", type="csv")
    #             if file:
    #                 df = pd.read_csv(file)
    #                 pr = df.profile_report()
    #                 st_profile_report(pr)
 
    #     with c2:
    #         pass

    elif tabs == 'Settings': #set the different settings for each AI tools {{"settings": "chatbot", "temperature:"},{},{}}
        colored_header(
        label="AI Parameters settings",
        description="Adjust the parameters to fine tune your AI engine",
        color_name="green-70",
        )
        pass
        if 'login_key' not in st.session_state or st.session_state.login_key == False:
            st.info("Data page accessible by educators only")
            st.error("Please log in to access this feature")
        elif st.session_state.login_key == True:
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
                        st.write("**:red[Class Tools Settings]**")
                        
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
                        st.write("**:red[Assess Myself Settings]**")
                        value = st.slider('Select the duration of the students recording in secs', 5, 30, 10, 1)             
                        submitted = st.form_submit_button("Submit")
                        if submitted:
                            user_info_collection.update_one({"tch_code": st.session_state.vta_code},{"$set": {"audio_settings": value}})
                            st.success("Changes Saved!")
                
                #st.write("Knowldege Map Settings")

                #st.write("Assess Myself Settings")
                #set duration


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
