# CherGPT
<img width="1050" alt="image" src="https://github.com/String-sg/chergpt2/assets/44336310/a8015029-46c5-45dc-a10e-559cad1d220d">
> [!WARNING]  
> This repository is no longer actively maintained. caa 10 Sep 2023, CherGPT is now in maintenance mode.
> Please see [CherGPT Starter Kit](https://github.com/String-sg/chergpt-starter-kit) instead 

<b>Main developer (Web)</b>: [Joe Tay](https://sg.linkedin.com/in/joe-tay2020)<br>
<b>Main developer (Telegram) and Web Starter Kit</b>: [Kahhow](https://sg.linkedin.com/in/leekahhow)<br>
Documentation previously maintained by [Kahhow](https://sg.linkedin.com/in/leekahhow) and Adrian<br>
Support and access previously managed by [Lance](https://sg.linkedin.com/in/lance-tan)*<be>

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
3) Deploy this app to streamlit. <br><br>
** If you would like to view chatlogs (basically a backend/ some sort of datastore). The basic chatbot setup without backend is simpler. More will be explained in the documentation (Work-in-progress)

You can develop your own function and call the function in the main(), you need to modify the default tab list in secrets.toml to add in a new function
Secrets.toml is part of your Streamlit deployment and is the equivalent of .env file
<br><br>
Note: Please refer to upcoming documentation on what to include in Secrets.toml. Should be just OpenAI API Key and also MongoDB URI (to be confirmed) 

```
db_host = "insert your own MongoDB URI"
OPENAI_API_KEY = "insert your own"
```
