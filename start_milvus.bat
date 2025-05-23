@echo off

echo Запуск Milvus сервера...

docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker не установлен. Установите Docker Desktop для запуска Milvus.
    pause
    exit /b 1
)

if not exist "milvus\db" mkdir milvus\db
if not exist "milvus\logs" mkdir milvus\logs
if not exist "milvus\wal" mkdir milvus\wal

docker run -d ^
  --name milvus-standalone ^
  --security-opt seccomp:unconfined ^
  -e ETCD_USE_EMBED=true ^
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd ^
  -e COMMON_STORAGETYPE=local ^
  -p 19530:19530 ^
  -p 9091:9091 ^
  -v "%cd%\milvus\db:/var/lib/milvus/db" ^
  -v "%cd%\milvus\logs:/var/lib/milvus/logs" ^
  -v "%cd%\milvus\wal:/var/lib/milvus/wal" ^
  milvusdb/milvus:v2.3.4 ^
  milvus run standalone

if %errorlevel% equ 0 (
    echo ✅ Milvus запущен на порту 19530
    echo 📊 Веб-интерфейс доступен на http://localhost:9091
    echo.
    echo Для остановки сервера выполните:
    echo docker stop milvus-standalone
    echo docker rm milvus-standalone
) else (
    echo ❌ Ошибка при запуске Milvus
)

pause
