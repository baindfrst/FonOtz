import nltk
import json
from rake_nltk import Rake
import pandas as pd
from nltk.corpus import stopwords
from yake import KeywordExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy3
def normalize_words(text):
    morph = pymorphy3.MorphAnalyzer()
    normalized_words = []
    for word in text.split():
        word = word.lower()
        word = word.strip('.,!?;:()[]{}"\'')
        p = morph.parse(word)[0]
        normalized_words.append(p.normal_form)
    return normalized_words

# Создаем объект Rake
rake = Rake()

# Задаем текст для извлечения ключевых слов
text = ("Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types")

# Извлекаем ключевые слова
rake.extract_keywords_from_text(text)
graph_from_rake = rake.graph_created_my

# Создаем пустой DataFrame
df = pd.DataFrame()

# Добавляем столбцы и строки в DataFrame
for k1, v1 in graph_from_rake.items():
    df[k1] = 0
    for k2, v2 in v1.items():
        df.loc[k2, k1] = v2

# Сохраняем DataFrame в виде таблицы в файл
df.to_excel('table.xlsx')

# Сохраняем DataFrame в виде таблицы в файл
df.to_csv('table.csv', index=False)

# Получаем список кортежей с ключевыми словами и их весами
keywords = rake.get_ranked_phrases_with_scores()

# Выводим ключевые слова на печать
for keyword, score in keywords:
    print(keyword, score)

#проверка алгоритма YAKE
text = "A conflict between language and atomistic information Fred Dretske and Jerry Fodor are responsible for popularizing three well-known theses in contemporary philosophy of mind: the thesis of Information-Based Semantics (IBS), the thesis of Content Atomism (Atomism) and the thesis of the Language of Thought (LOT). LOT concerns the semantically relevant structure of representations involved in cognitive states such as beliefs and desires. It maintains that all such representations must have syntactic structures mirroring the structure of their contents. IBS is a thesis about the nature of the relations that connect cognitive representations and their parts to their contents (semantic relations). It holds that these relations supervene solely on relations of the kind that support information content, perhaps with some help from logical principles of combination. Atomism is a thesis about the nature of the content of simple symbols. It holds that each substantive simple symbol possesses its content independently of all other symbols in the representational system. I argue that Dretske’s and Fodor’s theories are false and that their falsehood results from a conflict IBS and Atomism, on the one hand, and LOT, on the other"

extractor = KeywordExtractor(lan="eng",n=3,dedupLim=0.2,top=30)
data_result_yake_score = extractor.extract_keywords(text)
print(data_result_yake_score)