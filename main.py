from fastapi import FastAPI

app = FastAPI(title="Bare Boot Test")

@app.get("/__ping")
def __ping():
    return "OK"

@app.get("/health")
def health():
    return {"ok": True}