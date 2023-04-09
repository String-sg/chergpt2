
import streamlit as st
import pymongo
import csv
import hashlib
import certifi
from bson.objectid import ObjectId


# Connect to MongoDB
db_host = st.secrets["db_host"]
db_client = st.secrets["db_client"]
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]

# Define the list of databases and collections
databases = st.secrets["databases"]
collections = st.secrets["collections"]

# Define the CRUD functions
def create_document(db, collection, document):
    collection = db[collection]
    collection.insert_one(document)
    st.write("Document created successfully!")

def read_documents(db, collection):
    collection = db[collection]
    documents = list(collection.find({}))
    st.write("Documents:")
    for document in documents:
        st.write(document)

# def update_document(db, collection, document_id):
#     collection = db[collection]
#     document = collection.find_one({"_id": document_id})
#     if document:
#         st.write("Document:", document)
#         updated_document = {}
#         for key in document.keys():
#             updated_document[key] = st.text_input(key, value=str(document[key]))
#         if st.button("Update"):
#             collection.update_one({"_id": document_id}, {"$set": updated_document})
#             st.write("Document updated successfully!")
#     else:
#         st.write("Document not found.")

def update_document(db, collection_name, tch_code):
    collection = db[collection_name]
    document = collection.find_one({"tch_code": tch_code})
    if document:
        st.write("Document:", document)
        fields_to_delete = []
        for key, value in document.items():
            new_value = st.text_input(key, value=str(value))
            if st.button(f"Update {key}"):
                fields_to_delete.append(key)
            elif new_value != str(value):
                document[key] = new_value
        if fields_to_delete:
            collection.update_one({"tch_code": tch_code}, {"$unset": {field: "" for field in fields_to_delete}})
        if any(field != str(value) for field, value in document.items()):
            collection.update_one({"tch_code": tch_code}, {"$set": document})
            st.write("Document updated successfully!")
    else:
        st.write("Document not found.")

def delete_documents_by_id_code(db, collection_name, id_code):
    collection = db[collection_name]
    result = collection.delete_many({"$or": [{"tch_code": id_code}, {"vta_code": id_code}]})
    
    if result.deleted_count > 0:
        st.success(f"{result.deleted_count} document(s) deleted successfully.")
    else:
        st.warning("No documents found with the given id_code.")
    
    return result.deleted_count > 0

def delete_document(db, collection_name, tch_code):
    collection = db[collection_name]
    #document = collection.find_one({"_id": ObjectId(tch_code)})
    #collection.delete_one({"_id": ObjectId(tch_code)})
    result = collection.delete_one({"tch_code": tch_code})
    st.success("Document deleted successfully.")
    return result.deleted_count > 0

def delete_all_documents(db, collection):
    collection = db[collection]
    collection.delete_many({})
    st.write("All documents deleted successfully!")


def upload_csv(db, file):
    collection = db["user_info"]
    csvfile = file.read().decode('utf-8')
    reader = csv.DictReader(csvfile.splitlines())
    for row in reader:
        tch_code = row["tch_code"]
        api_key = row["api_key"]
        org = row["org"]
        profile = row["profile"]
        pass_key = row["pass_key"]
        tch_sub = row["tch_sub"]
        tch_level = row["tch_level"]

        # Hash the pass_key
        hashed_pass_key = hashlib.sha256(pass_key.encode()).hexdigest()

        # Create the document
        document = {
            "tch_code": tch_code,
            "api_key": api_key,
            "org": org,
            "profile": profile,
            "pass_key": hashed_pass_key,
            "tch_sub": tch_sub,
            "tch_level": tch_level
        }

        # Add vta_codes to the document
        for i in range(1, 46):
            vta_code_key = f"vta_code{i}"
            if vta_code_key in row:
                document[vta_code_key] = row[vta_code_key]

        # Insert the document into the collection
        collection.insert_one(document)

    st.write("CSV file uploaded and documents inserted successfully!")


def download_csv(db, collection):
    collection = db[collection]
    documents = list(collection.find({}))

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
        st.write("File is ready for downloading")
        st.download_button(
            label="Download data as CSV",
            data=data_csv,
            file_name=filename,
            mime='text/csv',
        )

def login(collection, username, password):
    # user_info_collection = db["user_info"]
    user = collection.find_one({"tch_code": username})
    if user:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if user["pass_key"] == hashed_password:
            st.write("Login successful!")
            user_profile = user.get("profile", None) #assign none if api key does not exist
            if user_profile == "administrator":
                return True
            else:
                st.error("Incorrect User Profile")
                return False
        else:
            st.error("Incorrect password!")
            return False
    else:
        st.error("User not found!")
        return False


def login_portal(collection):
   
    with st.form(key="authenticate"):
        st.write("Please key in your teacher code and password to access the data")
        teacher_code = st.text_input('Teacher code:')
        password = st.text_input('Password:', type="password")
        submit_button = st.form_submit_button(label='Login')
        if submit_button:
            if login(collection, teacher_code, password):
                st.session_state.login_key = True
                return True
            else:
                st.error("Incorrect teacher code or password")



def main():
    if 'login_key' not in st.session_state:
        st.session_state.login_key = False
    if 'db_key' not in st.session_state:
        st.session_state.db_key = False    

    if st.session_state.login_key == False:
        placeholder = st.empty()
        with placeholder:
            selected_database = st.sidebar.selectbox("Select a database", databases, index=0)
            selected_collection = st.sidebar.selectbox("Select a collection", collections, index=1, key='outside')
            st.session_state.db_key = selected_collection, selected_database
            db = client[selected_database]
            if login_portal(db[selected_collection]):
                st.session_state.login_key = True
                placeholder.empty() 
               
    
    if st.session_state.login_key == True:
        #placeholder.empty()
        
        # Show the CRUD options
        st.sidebar.title("CRUD Options")

        selected_collection, selected_database = st.session_state.db_key 
        st.sidebar.write(" Database is ", selected_database )
        db = client[selected_database]


        selected_collection = st.sidebar.selectbox("Select a collection", collections, index=1, key='inside')

        crud_option = st.sidebar.radio("", ["Create Document", "Read Documents", "Update Document", "Delete Document", "Delete All Documents", "Delete all by ID", "Upload CSV", "Download CSV", "Logout"])

        if crud_option == "Create Document":
            document = st.text_input("Enter the document")
            if st.button("Create"):
                create_document(db, selected_collection, document)

        if crud_option == "Read Documents":
            read_documents(db, selected_collection)

        if crud_option == "Update Document":
            tch_id = st.text_input("Enter the tch_id")
            #update = st.text_input("Enter the update")
            if st.button("Update"):
                update_document(db, selected_collection, tch_id)

        if crud_option == "Delete Document":
            document_id = st.text_input("Enter the tch_id")
            if st.button("Delete"):
                delete_document(db, selected_collection, document_id)


        if crud_option == "Delete all by ID":
            document_id = st.text_input("Enter the vta or tch code")
            if st.button("Delete"):
                delete_documents_by_id_code(db, selected_collection, document_id)

        if crud_option == "Delete All Documents":
            if st.button("Delete All"):
                delete_all_documents(db, selected_collection)

        if crud_option == "Upload CSV":
            file = st.file_uploader("Select a CSV file", type="csv")
            if file:
                upload_csv(db, file)

        if crud_option == "Download CSV":
            download_csv(db, selected_collection)


        if crud_option == "Logout":
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun() 

if __name__ == "__main__":
    main()
