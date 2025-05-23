import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np

from db import MilvusDB
from classifier import MessageClassifier

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise ValueError("Токен бота не найден! Убедитесь, что API_TOKEN указан в файле .env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

try:
    db = MilvusDB()
    classifier = MessageClassifier()
    logger.info("База данных и классификатор инициализированы успешно")
except Exception as e:
    logger.error(f"Ошибка инициализации: {e}")
    db = None
    classifier = None

class BotStates(StatesGroup):
    waiting_for_message = State()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    welcome_text = """
🤖 Добро пожаловать в бота классификации сообщений!

Я умею:
• 📝 Классифицировать ваши сообщения по темам
• 🏷️ Автоматически присваивать теги
• 💾 Сохранять сообщения в векторную базу данных
• 🔍 Искать похожие сообщения

Просто отправьте мне любое сообщение, и я его обработаю!

Команды:
/start - показать это сообщение
/stats - статистика базы данных
/search - поиск похожих сообщений
/help - помощь
"""
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Статистика", callback_data="stats")],
        [InlineKeyboardButton(text="🔍 Поиск", callback_data="search")],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="help")]
    ])
    
    await message.answer(welcome_text, reply_markup=keyboard)

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    if not db:
        await message.answer("❌ База данных недоступна")
        return
    
    try:
        stats = db.get_stats()
        stats_text = f"""
📊 **Статистика базы данных:**

📝 Всего сообщений: {stats['total_messages']}
👤 Ваш ID: {message.from_user.id}
🕐 Время запроса: {datetime.now().strftime('%H:%M:%S')}
"""
        await message.answer(stats_text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        await message.answer("❌ Ошибка при получении статистики")

@dp.message(Command("search"))
async def cmd_search(message: Message, state: FSMContext):
    if not db or not classifier:
        await message.answer("❌ Сервисы недоступны")
        return
    
    await message.answer("🔍 Отправьте сообщение для поиска похожих записей:")
    await state.set_state(BotStates.waiting_for_message)

@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """
🆘 **Справка по использованию бота:**

**Автоматическая классификация:**
Просто отправьте любое текстовое сообщение, и бот:
1. Определит его тему (работа, учеба, семья, здоровье, досуг, финансы, важное)
2. Создаст векторное представление
3. Сохранит в базу данных

**Доступные темы:**
• 💼 Работа - проекты, встречи, задачи
• 📚 Учеба - лекции, экзамены, домашние задания
• 👨‍👩‍👧‍👦 Семья - встречи с родными, семейные дела
• 🏥 Здоровье - врачи, спорт, самочувствие
• 🎮 Досуг - развлечения, хобби, отдых
• 💰 Финансы - покупки, счета, бюджет
• ⚠️ Важное - срочные и приоритетные дела

**Команды:**
/start - главное меню
/stats - статистика
/search - поиск похожих сообщений
/help - эта справка

**Поиск:**
Используйте команду /search, затем отправьте текст для поиска похожих сообщений в базе данных.
"""
    await message.answer(help_text, parse_mode="Markdown")

@dp.callback_query()
async def process_callback(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "stats":
        await cmd_stats(callback_query.message)
    elif callback_query.data == "search":
        await cmd_search(callback_query.message, state)
    elif callback_query.data == "help":
        await cmd_help(callback_query.message)
    
    await callback_query.answer()

@dp.message(BotStates.waiting_for_message)
async def process_search_query(message: Message, state: FSMContext):
    if not message.text:
        await message.answer("❌ Пожалуйста, отправьте текстовое сообщение")
        return
    
    if not db or not classifier:
        await message.answer("❌ Сервисы недоступны")
        await state.clear()
        return
    
    try:
        query_vector = classifier.get_text_embedding(message.text)
        similar_messages = db.search_similar(query_vector, limit=5)
        
        if not similar_messages:
            await message.answer("🔍 Похожие сообщения не найдены")
        else:
            response = "🔍 **Похожие сообщения:**\n\n"
            for i, msg in enumerate(similar_messages, 1):
                similarity = (1 - msg['distance']) * 100
                response += f"**{i}.** `{msg['text'][:100]}...`\n"
                response += f"   🏷️ Тег: {msg['tags']}\n"
                response += f"   📊 Схожесть: {similarity:.1f}%\n"
                response += f"   📅 Дата: {datetime.fromtimestamp(msg['created_time']).strftime('%d.%m.%Y %H:%M')}\n\n"
            
            await message.answer(response, parse_mode="Markdown")
    
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        await message.answer("❌ Ошибка при поиске")
    
    await state.clear()

@dp.message()
async def handle_message(message: Message):
    if not message.text:
        await message.answer("❌ Я работаю только с текстовыми сообщениями")
        return
    
    if not db or not classifier:
        await message.answer("❌ Сервисы недоступны. Попробуйте позже.")
        return
    
    processing_msg = await message.answer("⏳ Обрабатываю сообщение...")
    
    try:
        tag = classifier.classify_message(message.text)
        vector = classifier.get_text_embedding(message.text)
        success = db.insert_message(
            text=message.text,
            vector=vector,
            tags=tag,
            user_id=message.from_user.id
        )
        
        if success:
            similar_tags = classifier.get_similar_tags(message.text, top_k=3)
            
            response = f"""
✅ **Сообщение обработано и сохранено!**

📝 **Текст:** `{message.text[:100]}...` 

🏷️ **Присвоенный тег:** `{tag}`

📊 **Другие возможные теги:**
"""
            for i, (tag_name, probability) in enumerate(similar_tags, 1):
                response += f"{i}. {tag_name} ({probability:.1%})\n"
            
            response += f"\n💾 **Статус:** Сохранено в базе данных"
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔍 Найти похожие", callback_data="search")],
                [InlineKeyboardButton(text="📊 Статистика", callback_data="stats")]
            ])
            
            await processing_msg.edit_text(response, parse_mode="Markdown", reply_markup=keyboard)
        else:
            await processing_msg.edit_text("❌ Ошибка при сохранении в базу данных")
    
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}")
        await processing_msg.edit_text("❌ Произошла ошибка при обработке сообщения")

async def main():
    logger.info("Запускаем бота...")
    
    if not db or not classifier:
        logger.warning("Некоторые сервисы недоступны. Бот запустится с ограниченной функциональностью.")
    
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()

if __name__ == '__main__':
    asyncio.run(main())
