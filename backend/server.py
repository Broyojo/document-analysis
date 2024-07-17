from fastapi import FastAPI

from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/")
async def root():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())
