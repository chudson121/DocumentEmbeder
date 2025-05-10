# DocumentEmbeder

I want to take a folder of files create the embeddings for them and then use them during RAG prompts.

Requirements
I would like to only do embeddings if the file is new or modified since the last run time.
Before the prompt is sent to the LLM, search through the vectorDB and retrieve semantically similar responses, and add that information to the LLM request.
return answer

## Technical Details

### Install UV
https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv
```python

**Self installers** 

curl -LsSf https://astral.sh/uv/install.sh | sh
wget -qO- https://astral.sh/uv/install.sh | sh
uv self update

** Package installers **
winget install --id=astral-sh.uv  -e
pip install uv
pip install --upgrade uv

**Common Commands**
uv --version
uv python install

# reinstall
uv python install --reinstall 

# specific version 
uv venv --python 3.12.0 

uv python list

uv init
uv venv

uv run .\DocumentRetrieve.py
```

### venv
```python
python.exe -m pip install --upgrade pip

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade langchain
pip install --upgrade langchain

#add -O to turn debug off
#ctlr k, ctrl c to comment
```

### Setup for deployment

```python
# Generate requirements.txt
pip freeze > requirements.txt
pip install -r requirements.txt

```



```powershell
ps to stream a file
Get-Content "$($env:USERPROFILE)\AppData\Local\Ollama\server.log" -wait
```
