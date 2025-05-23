import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List, Tuple
import re

logger = logging.getLogger(__name__)

class MessageClassifier:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tag_classifier = self._setup_tag_classifier()
        logger.info("Классификатор инициализирован")
    
    def _setup_tag_classifier(self) -> Pipeline:
        train_texts = [
            'работаю над проектом', 'встреча с коллегами', 'deadline приближается',
            'отчет по работе', 'презентация завтра', 'обсуждение с командой',
            'совещание в офисе', 'задача от руководителя', 'проект почти готов',
            
            'лекция по математике', 'экзамен завтра', 'домашнее задание',
            'курсовая работа', 'семинар в университете', 'готовлюсь к зачету',
            'библиотека закрыта', 'конспект по физике', 'сдача лабораторной',
            
            'встреча с родителями', 'день рождения сестры', 'семейный ужин',
            'поездка с детьми', 'помощь маме', 'звонок от бабушки',
            'семейные выходные', 'забрать ребенка', 'готовим вместе',
            
            'запись к врачу', 'болит голова', 'поход в спортзал',
            'диета и питание', 'прием витаминов', 'медицинский осмотр',
            'зубной врач', 'анализы готовы', 'самочувствие улучшилось',
            
            'просмотр фильма', 'встреча с друзьями', 'поход в кино',
            'чтение книги', 'игра в футбол', 'концерт любимой группы',
            'путешествие на выходные', 'новая видеоигра', 'прогулка в парке',
            
            'покупка продуктов', 'оплата счетов', 'накопления на отпуск',
            'банковская карта', 'инвестиции в акции', 'бюджет на месяц',
            'кредитный платеж', 'покупка машины', 'страховка дома',
            
            'срочное сообщение', 'важная новость', 'критическая ситуация',
            'неотложное дело', 'приоритетная задача', 'экстренный вызов',
            'внимание всем', 'требует немедленного решения', 'чрезвычайная ситуация'
        ]
        
        train_labels = (
            ['работа'] * 9 + 
            ['учеба'] * 9 + 
            ['семья'] * 9 + 
            ['здоровье'] * 9 + 
            ['досуг'] * 9 + 
            ['финансы'] * 9 + 
            ['важное'] * 9
        )
        
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words=None)),
            ('nb', MultinomialNB())
        ])
        
        classifier.fit(train_texts, train_labels)
        return classifier
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        try:
            cleaned_text = self._clean_text(text)
            embedding = self.sentence_model.encode(cleaned_text)
            return embedding
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}")
            return np.zeros(384)
    
    def classify_message(self, text: str) -> str:
        try:
            cleaned_text = self._clean_text(text)
            tag = self.tag_classifier.predict([cleaned_text])[0]
            probabilities = self.tag_classifier.predict_proba([cleaned_text])[0]
            max_prob = max(probabilities)
            if max_prob < 0.3:
                tag = "разное"
            return tag
        except Exception as e:
            logger.error(f"Ошибка при классификации: {e}")
            return "разное"
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        return text.lower()
    
    def get_similar_tags(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        try:
            cleaned_text = self._clean_text(text)
            probabilities = self.tag_classifier.predict_proba([cleaned_text])[0]
            classes = self.tag_classifier.classes_
            tag_probs = list(zip(classes, probabilities))
            tag_probs.sort(key=lambda x: x[1], reverse=True)
            return tag_probs[:top_k]
        except Exception as e:
            logger.error(f"Ошибка при получении похожих тегов: {e}")
            return [("разное", 1.0)]
