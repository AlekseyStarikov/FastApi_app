from loguru import logger

# Настройка логирования
logger.add("logs/app.log", rotation="500 KB", retention="10 days", level="INFO")