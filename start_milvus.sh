#!/bin/bash


echo "Запуск Milvus сервера..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker для запуска Milvus."
    exit 1
fi

mkdir -p ./milvus/db
mkdir -p ./milvus/logs
mkdir -p ./milvus/wal

docker run -d \
  --name milvus-standalone \
  --security-opt seccomp:unconfined \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e COMMON_STORAGETYPE=local \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus/db:/var/lib/milvus/db \
  -v $(pwd)/milvus/logs:/var/lib/milvus/logs \
  -v $(pwd)/milvus/wal:/var/lib/milvus/wal \
  milvusdb/milvus:v2.3.4 \
  milvus run standalone

echo "✅ Milvus запущен на порту 19530"
echo "📊 Веб-интерфейс доступен на http://localhost:9091"
echo ""
echo "Для остановки сервера выполните:"
echo "docker stop milvus-standalone"
echo "docker rm milvus-standalone"
