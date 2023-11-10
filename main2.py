from fastapi import FastAPI

app=FastAPI()

@app.get("/countries/")
async def country(country_name: str="japan",country_no: int=1):
    return {
        "country_name":country_name,
        "country_no":country_no}