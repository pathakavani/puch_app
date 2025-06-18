# puch.ai mcp server application


## > my implementation for the Puch.ai job application process
this was such a refreshing and engaging way to apply for a position! instead of the usual resume submission, i got to build an actual working MCP server - more fun and gave a chance to explore my tech skills.


## what I did
### 1. setup
```bash
# created virtual environment from requirements
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. resume
- placed my `resume.pdf` in the same directory as the server
- the server automatically detects and converts it to markdown

### 3. making my mcp server public
leveraged serveo.net for quick public access:
```bash
python mcp_server.py  # start server on localhost:8085
ssh -R 80:localhost:8085 serveo.net  # make it public
```

### 4. config
    updated the starter code with:
    - my application key from `/apply` command
    - phone number in `{country_code}{number}` format

    then connected via:
    ``` bash
    /mcp connect <serveo_url>/mcp <auth_token>
    ```

## key files
- `mcp_server.py` - Enhanced starter code with working resume tool
- `requirements.txt` - All necessary dependencies
- `resume.pdf` - My resume file

---

**RESULT**: a working MCP server that Puch.ai can validate and retrieve my resume from. much more interesting than traditional applications! 🚀