
#Check completed - need to upload pdf to mongodb
import streamlit as st
import pandas as pd
import pymongo
import certifi
import pypdf
import io
import os
import gridfs
from pypdf import PdfReader
from st_aggrid import AgGrid
import configparser
import ast
from bson import ObjectId
from datetime import datetime
from data_process import generate_resource


config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())

db = client[db_client]
fs = gridfs.GridFS(db)
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
doc_collection = db[config['constants']['dd']]

subjects_str = config.get('menu_lists', 'subjects')
shared_access_str = config.get('menu_lists', 'shared_access')
subjects = ast.literal_eval(subjects_str)
shared_access = ast.literal_eval(shared_access_str)



def extract_first_n_words(text, n=200):
	words = text.split()
	first_n_words = words[:n]
	return " ".join(first_n_words)

def extract_text_from_pdf(uploaded_file):
	"""
	Extract text from the first page of an uploaded PDF file.
	:param uploaded_file: Streamlit UploadedFile object
	:return: Extracted text as a single string
	"""
	# Store the PDF file in memory
	in_memory_pdf = io.BytesIO(uploaded_file.getvalue())
	
	# Initialize the PDF reader
	pdf_reader = PdfReader(in_memory_pdf)
	
	# Extract text from the first page
	first_page = pdf_reader.pages[0]
	pdf_text = first_page.extract_text()
	
	return pdf_text

# Delete a document by ID along with its corresponding PDF file
def delete_document_by_id_with_file(document_id):
	try:
		# Query the document by its ID
		document = doc_collection.find_one({"_id": ObjectId(document_id)})

		if document is not None:
			# Check if the tch_code of the document matches the st.session_state.vta_code
			if document["tch_code"] == st.session_state.vta_code:
				# Delete the corresponding PDF file from GridFS
				file_id = document.get('file_id')
				if file_id:
					fs.delete(file_id)

				# Delete the document from MongoDB
				result = doc_collection.delete_one({"_id": ObjectId(document_id)})
			   
				return result.deleted_count > 0
			else:
				st.error("You can only delete your own resources.")
				return False
		else:
			st.error("Document not found.")
			return False
	except Exception as e:
		st.write(f"Error: {e}")
		return False


def upload_pdf(file):
	return fs.put(file, filename=file.name, contentType="application/pdf")

def is_document_duplicate(new_content, tch_code, doc_collection):
	for doc in doc_collection.find({"tch_code": tch_code}):
		if doc['content'] == new_content:
			return True
	return False



def upload_resource():

	uploaded_file = st.file_uploader("Choose a file to upload as a resource (Only your first 10 uploaded articles will be used)")
	if uploaded_file is not None:
		txt = extract_text_from_pdf(uploaded_file)

	
		with st.form("Tool Settings"):
			st.write("**:blue[Please fill in the details as accurately as possible so that the AI agent could locate your sources]**")
			#st.warning("""**:red[(Note that you can only upload up to 10 resources)]**""")
			 

			# Create a text input for subject
			subject = st.selectbox("Subject", options=subjects)

			# Create a text input for topic
			topic = st.text_input("Topic: (50 characters)", max_chars=50)
			# Create a text input for source
			source = st.text_input("Document Source", max_chars=50)

			# Create a text input for topic description
			hyperlinks = st.text_input("Enter a hyperlink (YouTube / Webpage ): (Max 250 characters)", max_chars=200)

			# Create a text area input
			#st.write(":blue[Please confirmed that this is the resource you wish to upload: ( First 200 words shown)]")
			st.text_area(":blue[Please confirmed that this is the resource you wish to upload: ( First 200 words shown)]",value=extract_first_n_words(txt, n=200), height=300)
			#stx.scrollableTextbox(txt, height=500, border=True)
			#st.write(txt)
			st.write("#")

			# Add a multiselect input for class access
			class_access = st.selectbox("Select resource access (Note: Shared resource allow sharing across schools):", options=shared_access)


			# Create a submit button
			submit_button = st.form_submit_button("Submit")

			# Handle form submission
			if submit_button:
				if not source or not topic:
					st.warning("Source and Topic cannot be left empty. Please fill in the required fields.")
					return
				content = txt
				tch_code = st.session_state.vta_code

				# Check if the document is a duplicate for the same teacher
				if is_document_duplicate(content, tch_code, doc_collection):
					st.warning("This document has already been uploaded by you. Please upload a different document.")
					return

				#cache_api()
				content = txt
				tch_code = st.session_state.vta_code
				file_id = upload_pdf(uploaded_file)
				#st.success(f"File uploaded with ID: {file_id}")
				
				# Create a dictionary to store the document data
				document_data = {
					"subject": subject,
					"topic": topic,
					"source": source,
					"hyperlinks": hyperlinks,
					"content": content,
					"tch_code": tch_code,
					"class_access": class_access,
					"file_id": file_id
				}

				# Insert the document data into the MongoDB collection
				doc_collection.insert_one(document_data)
				st.success("Document uploaded successfully!")



def include_document_for_class_resource(tch_code, doc_collection):
	with st.form(key="include_resource_form"):
	# Ask the teacher to input a document ID
		document_id = st.text_input("Please enter the Document ID you would like to include for the class resources: ")

		
		submit_button = st.form_submit_button("Include Resource")

		# Handle form submission
		if submit_button:
		# Check if the document is available and if its class_access says 'Shared resource'
			# Search for the document in the doc_collection
			if not document_id:
				st.warning("Document ID cannot be left empty. Please fill in the required fields.")
				return
			document = doc_collection.find_one({"_id": ObjectId(document_id)})
			if document and document.get("class_access") == "Shared resource":

				# Update or create the tch_sharing_resource field
				if "tch_sharing_resource" not in document:
					document["tch_sharing_resource"] = [tch_code]
				else:
					document["tch_sharing_resource"].append(tch_code)

				# Update the doc_collection with the modified document
				doc_collection.update_one({"_id": document_id}, {"$set": {"tch_sharing_resource": document["tch_sharing_resource"]}})

				# Copy the entire document without the tch_sharing_resource field
				copied_document = document.copy()
				copied_document.pop("tch_sharing_resource")
				copied_document.pop("_id")  # Remove the _id field

				# Update the tch_code and class_access fields
				copied_document["tch_code"] = tch_code
				copied_document["class_access"] = "Copied Resource"

				# Insert the copied document into the doc_collection
				doc_collection.insert_one(copied_document)

				st.success("Document successfully included for the class resources.")
			else:
				st.error("Document not found or not a shared resource.")


def check_directory_exists():
	if 'admin_key' not in st.session_state:
		st.session_state.admin_key = False 

	with st.form(key="Check Directory and Generate Database"):
		directory_path = os.path.join(os.getcwd(), st.session_state.vta_code)

		if os.path.exists(directory_path):
			user_info = user_info_collection.find_one({"tch_code": st.session_state.vta_code})

			if user_info and "db_last_created" in user_info:
				st.success(f"Document database exists, last created on {user_info['db_last_created']}. Please see the information below for your current database")
			if "db_subject" in user_info and "db_description" in user_info:
			   st.write(f'Database Subject: {user_info["db_subject"]}')
			   st.write(f'Database Subject: {user_info["db_description"]}')
			else:
				st.warning("Database exists, but subject and description not found in the database")

		else:
			st.write("No document database is created.")

		st.info("Please regenerate a new database if you have remove or add document resources above")
		# Add inputs for description and topic
		db_description = st.text_input("Enter a description for the database (max 100 characters, example given):", max_chars=100, value="Relevant resources and materials shared by your class teacher to enhance your learning experience and deepen your understanding of the subject.")
		db_subject = st.selectbox("Select the subject", options=subjects)

		generate_button = st.form_submit_button("Generate Database")

		if generate_button:
			if db_description and db_subject:
				st.info(generate_resource(st.session_state.vta_code, st.session_state.admin_key))
				now = datetime.now()
				formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
				user_info_collection.update_one(
					{"tch_code": st.session_state.vta_code},
					{"$set": {"db_last_created": formatted_now, "db_description": db_description, "db_subject": db_subject}}
				)
				st.success("Database generated successfully. Please click dashboard to refresh")
			else:
				st.warning("Please provide both a description and a topic.")


def dashboard():

	col1, col2 = st.columns([2,2])
	with col1:

		st.info("Current student and interaction data (Filter the data and right click to download in CSV or Excel format)")
		try:
			codes = st.session_state.codes_key + [st.session_state.vta_code]
			#st.write(codes)
			documents = list(data_collection.find({"vta_code": {"$in": codes }}, {'_id': 0}))
			#documents = list(doc_collection.find({"class_access": "Shared resource"}, {'_id': 0}))
			df = pd.DataFrame(documents)
			#aggrid_interactive_table(df=df)
			AgGrid(df, height='400px', key="data")
		except Exception as e:
			st.write(f"Error: {e}")
			return False

	with col2:
		# ["Shared resource","School resource", "Class resource"]

		st.info("Resources available that is available for sharing")
		try:
			# Retrieve documents including the _id column
			documents = list(doc_collection.find({"class_access": "Shared resource"}, {'file_id': 0}))
			if documents:
				# Create a DataFrame from the documents
				r_df = pd.DataFrame(documents)

				# Convert the _id column to a string
				r_df['_id'] = r_df['_id'].astype(str)

				# Rename the _id column to 'Document ID'
				r_df = r_df.rename(columns={'_id': 'Document ID'})
				AgGrid(r_df, height='400px', key="global_resources")
			else:
				df = pd.DataFrame(documents)
				#aggrid_interactive_table(df=df)
				AgGrid(df, height='400px', key="no_elements")
		except Exception as e:
			st.write(f"Error: {e}")
			return False

	
	col1, col2 = st.columns([2,2])
	with col1:
		st.warning("Add shared documents to class resource")
		include_document_for_class_resource(st.session_state.vta_code, doc_collection)
		st.warning("Upload PDF resources")
		upload_resource()

	with col2:
		st.warning(f"Resources uploaded or included by teacher : {st.session_state.vta_code}")
		try:
			# Retrieve documents including the _id column
			 # Retrieve documents including the _id column
			documents = list(doc_collection.find({"tch_code": st.session_state.vta_code}, {'file_id': 0}))
			if len(documents) == 0:
				st.info("No documents found.")
			else:
				# Create a DataFrame from the documents
				r_df = pd.DataFrame(documents)

				# Convert the _id column to a string
				r_df['_id'] = r_df['_id'].astype(str)

				# Rename the _id column to 'Document ID'
				r_df = r_df.rename(columns={'_id': 'Document ID'})
				AgGrid(r_df, height='400px',key="class_resources")
		except Exception as e:
			st.write(f"Error: {e}")
			return False

		st.warning(f"IMPORTANT: After uploading or adding shared documents, please generate your document database for your students to access!")
		check_directory_exists()

		st.error("Delete a document resource")
		with st.form(key="delete_form"):
			# Create a text input for the document ID
			
			document_id = st.text_input("Enter the Document ID (copy paste)to delete your resource:")
			st.error("You are about to delete the above resource! To proceed please click the delete button")
			# Create a submit button
			submit_button = st.form_submit_button("Delete")

			# Handle form submission
			if submit_button:
				if delete_document_by_id_with_file(document_id):
					st.success(f"Document with ID '{document_id}' deleted successfully.(Click Dashboard to refresh)")
				else:
					st.warning(f"Failed to delete document with ID '{document_id}'.")





