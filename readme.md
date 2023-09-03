# CherGPT v2

Developed by Joe Tay<br>
Documentation maintained by Kahhow and Adrian<br>
Support and access managed by Lance<br>

## Setup instructions
In your terminal, run:<br>
```pip install -r requirements.txt```
<br><br>
Please approach the String Team for secrets.toml credentials to be defined under .streamlit folder
<br><br>
Thereafter, in your terminal, run:<br>
```streamlit run main.py```

### Required Setup
1) You need to open a MongoDB account 
2) Retrieve your own OpenAPI key 
3) Deploy this app to streamlit. 

You can develop your own function and call the function in the main(), you need to modify the default tab list in secrets.toml to add in a new function
Secrets.toml is part of your Streamlit deployment and is the equivalent of .env file
<br><br>
Note: please approach a member of the String team for assistance with secrets.toml 

```
db_host = "insert your own MongoDB URI"
OPENAI_API_KEY = "insert your own"
```
