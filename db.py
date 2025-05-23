import random
import time
import logging
import os
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from typing import List, Dict, Any
import numpy as np

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusDB:
    def __init__(self, host=None, port=None, collection_name=None):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "history_docs")
        self.collection = None
        self.connect()
        self.setup_collection()
    
    def connect(self):
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info(f"Подключено к Milvus на {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Milvus: {e}")
            raise
    
    def setup_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="created_time", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(fields, description="Telegram bot messages with classification")
        
        if utility.has_collection(self.collection_name):
            logger.info(f"Коллекция {self.collection_name} уже существует")
            self.collection = Collection(name=self.collection_name)
        else:
            logger.info(f"Создаем новую коллекцию {self.collection_name}")
            self.collection = Collection(name=self.collection_name, schema=schema)
        
        self.create_index()
        self.collection.load()
    
    def create_index(self):
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        if not self.collection.has_index():
            self.collection.create_index(field_name="vector", index_params=index_params)
            logger.info("Индекс создан")
    
    def insert_message(self, text: str, vector: np.ndarray, tags: str, user_id: int) -> bool:
        try:
            data = [
                [vector.tolist()],
                [text],
                [tags],
                [user_id],
                [int(time.time())]
            ]
            
            mr = self.collection.insert(data)
            self.collection.flush()
            logger.info(f"Сообщение вставлено с ID: {mr.primary_keys}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при вставке сообщения: {e}")
            return False
    
    def search_similar(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=["text", "tags", "user_id", "created_time"]
            )
            
            similar_messages = []
            for hits in results:
                for hit in hits:
                    similar_messages.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "text": hit.entity.get("text"),
                        "tags": hit.entity.get("tags"),
                        "user_id": hit.entity.get("user_id"),
                        "created_time": hit.entity.get("created_time")
                    })
            
            return similar_messages
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []
    
    def get_stats(self) -> Dict[str, int]:
        try:
            self.collection.flush()
            num_entities = self.collection.num_entities
            return {"total_messages": num_entities}
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            return {"total_messages": 0}


