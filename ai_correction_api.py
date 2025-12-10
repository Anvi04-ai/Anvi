from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import BytesIO
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import process
import pycountry, geonamescache, torch

app = FastAPI(title="CleanChain AI Correction Engine")

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def root():
    return {"message": "AI Correction Engine is running!"}

@app.post("/correct_file")
async def correct_file(file: UploadFile = File(...)):
    df = pd.read_excel(BytesIO(await file.read())) if file.filename.endswith(".xlsx") else pd.read_csv(file.file)

    countries = [c.name for c in pycountry.countries]
    gc = geonamescache.GeonamesCache()
    cities = [c["name"] for c in gc.get_cities().values()]

    def correct_name(name, ref_list):
        matches = process.extractOne(name, ref_list)
        if matches and matches[1] > 80:
            return matches[0]
        return name

    if "country" in df.columns:
        df["country"] = df["country"].apply(lambda x: correct_name(str(x), countries))
    if "city" in df.columns:
        df["city"] = df["city"].apply(lambda x: correct_name(str(x), cities))

    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return JSONResponse({"message": "File corrected", "rows": len(df)})
