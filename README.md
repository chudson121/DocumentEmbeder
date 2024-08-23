# DocumentEmbeder


# Generate venv
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade langchain
pip install --upgrade langchain

# Generate requirements.txt
pip freeze > requirements.txt
pip install -r requirements.txt


#add -O to turn debug off
#ctlr k, ctrl c to comment

ps to stream a file
Get-Content "$($env:USERPROFILE)\AppData\Local\Ollama\server.log" -wait