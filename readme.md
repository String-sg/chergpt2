# CherGPT Starter kit
<img width="689" alt="image" src="https://github.com/String-sg/chergpt2/assets/44336310/589dae66-8345-4ae5-9d4f-780cfec138e5">

<b>Main developer (Web)</b>: Joe Tay<br>
<b>Main developer (Telegram) and Web Starter Kit</b>: Kahhow<br>
Documentation maintained by Kahhow and Adrian<br>
Support and access managed by Lance*<be>

*caa 10 Sep 2023, CherGPT is now in maintenance mode. The team is working on 'CherGPT Starter Kit' to help more educators deploy their own AI powered chatbot. 

## Setup instructions
In your terminal, run:<br>
```pip install -r requirements.txt```
<br><br>
Please approach the String Team for secrets.toml credentials to be defined under .streamlit folder
<br><br>
Thereafter, in your terminal, run:<br>
```streamlit run main.py```

### Required Setup
1) MongoDB account** 
2) Retrieve your own OpenAPI key 
3) Deploy this app to streamlit. 
** If you would like to view chatlogs (basically a backend/ some sort of datastore). The basic chatbot setup without backend is simpler. More will be explained in the documentation (Work-in-progress)

You can develop your own function and call the function in the main(), you need to modify the default tab list in secrets.toml to add in a new function
Secrets.toml is part of your Streamlit deployment and is the equivalent of .env file
<br><br>
Note: Please refer to upcoming documentation on what to include in Secrets.toml. Should be just OpenAI API Key and also MongoDB URI (to be confirmed) 

```
db_host = "insert your own MongoDB URI"
OPENAI_API_KEY = "insert your own"
```
