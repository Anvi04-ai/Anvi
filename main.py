from fastapi import FastAPI, UploadFile, File
import pandas as pd
from fuzzywuzzy import fuzz
from textblob import Word

app = FastAPI(title="CleanChain AI", description="B2B Data Cleaning API", version="0.1")

@app.post("/clean")
async def clean_data(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Capitalize and strip whitespace
    df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)

    # Remove duplicates
    df = df.drop_duplicates()

    # Optional: Basic spelling correction
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: str(Word(x).correct()) if isinstance(x, str) else x)

    # Save to new cleaned CSV
    cleaned_file = "cleaned_output.csv"
    df.to_csv(cleaned_file, index=False)

    return {"message": "File cleaned successfully!", "cleaned_rows": len(df)}
