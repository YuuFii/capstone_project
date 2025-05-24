from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "sentra_db")

def get_mongo_client():
    client = MongoClient(MONGO_URI)
    return client

def get_collection(collection_name):
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[collection_name]
    return collection