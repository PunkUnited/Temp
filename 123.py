#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
import heapq
from collections import defaultdict
from json import loads
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Lambda,
    Dropout, BatchNormalization, Flatten, Embedding, Masking, Attention, Subtract, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # Разрешить рост использования GPU-памяти по мере необходимости
config.log_device_placement = True  # Вывести информацию о размещении операций на устройствах

# Установить настройку для использования всей доступной оперативной памяти
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.intra_op_parallelism_threads = tf.config.threading.get_inter_op_parallelism_threads()  # Использовать все доступные потоки CPU для операций внутри сеанса TensorFlow
config.inter_op_parallelism_threads = tf.config.threading.get_inter_op_parallelism_threads()  # Использовать все доступные потоки CPU для операций между сеансами TensorFlow
# Создание сеанса TensorFlow с указанной конфигурацией
session = tf.compat.v1.Session(config=config)

# Установка созданного сеанса TensorFlow в качестве глобального сеанса по умолчанию
tf.compat.v1.keras.backend.set_session(session)



def build_siamese_network(cat_dicts, max_imgs_all, num_filters_images=64, kernel_size=4, dense_units_for_extracting=512, dense_units_for_extracting_main_image=128, embedding_dim=32, dropout_coef=0.2, activation='relu', out_layer=2, func="diff", regular=0.01):
    """
    Функция для построения сети Siamese.

    Аргументы:
    Обязательные
    - cat_dicts: словарь, содержащий информацию о категориях
    - max_imgs_all: максимальное количество изображений
    Необязательные
    - num_filters_images: количество фильтров для обработки изображений
    - kernel_size: размер ядра свертки
    - dense_units_for_extracting: количество нейронов в плотном слое для извлечения признаков
    - dense_units_for_extracting_main_image: количество нейронов в плотном слое для извлечения признаков главного изображения
    - embedding_dim: размерность пространства вложения
    - dropout_coef: коэффициент отсева для слоев Dropout
    - activation: функция активации
    - out_layer: количество выходных нейронов
    - func: функция расстояния ('diff', 'cos_sim', 'prod', 'euclid', 'concat')

    Возвращает:
    - siamese_network: модель сети Siamese
    """

    # Входные слои
    vector_images_all_input = Input(shape=(max_imgs_all, 128))
    vector_main_image_input = Input(shape=(128))
    bert_vector_input = Input(shape=(64,))

    # Списки слоев для каждой категории
    input_layers = []
    mask_layers = []
    embedding_layers = []
    flatten_layers = []

    # Получение списка категорий
    cat_columns = list(cat_dicts.keys())

    # Создание слоев для каждой категории
    for column in cat_columns:
        # Количество уникальных категорий в столбце
        num_categories = len(cat_dicts[column]) + 1

        # Создание входного слоя для текущей категории
        input_layer = Input(shape=(1,))
        input_layers.append(input_layer)

        # Создание слоя маскирования
        mask_layer = Masking(mask_value=0)(input_layer)
        mask_layers.append(mask_layer)

        # Создание слоя вложения
        embedding_layer = Embedding(num_categories, embedding_dim, embeddings_regularizer=l2(l=regular))(mask_layer)
        embedding_layers.append(embedding_layer)

        # Создание слоя выравнивания
        flatten_layer = Flatten()(embedding_layer)
        flatten_layers.append(flatten_layer)
    if len(flatten_layers) != 0:
        flatten_layers[-1] = Attention()([flatten_layers[-1], flatten_layers[-1]])
        
        
    # Обработка векторов изображений
    mask_layer_images = Masking(mask_value=0.0)(vector_images_all_input)
    conv1d_all_images = Conv1D(filters=num_filters_images, kernel_size=kernel_size, activation='relu')(mask_layer_images)
    max_pool_all_images = GlobalMaxPooling1D()(conv1d_all_images)

    # Обработка главного изображения
    dense_main_1 = Dense(units=dense_units_for_extracting_main_image, activation='relu')(vector_main_image_input)
    dropout_main_1 = Dropout(rate=dropout_coef)(dense_main_1)
    dense_main_2 = Dense(units=dense_units_for_extracting_main_image // 2, activation='relu')(dropout_main_1)
    dropout_main_2 = Dropout(rate=dropout_coef)(dense_main_2)
    dense_main_3 = Dense(units=dense_units_for_extracting_main_image//8, activation='relu')(dropout_main_2)
    dropout_main_3 = Dropout(rate=dropout_coef)(dense_main_3)

    # Соединение слоев обработки всех данных
    merged_inputs = Concatenate()([*flatten_layers, max_pool_all_images, dropout_main_3, bert_vector_input])


    # Обработка объединенных входов
    dense_all_1 = Dense(units=dense_units_for_extracting, activation='relu')(merged_inputs)

    dense_all_2 = Dense(units=dense_units_for_extracting // 2, activation='relu')(dense_all_1)
    dense_all_3 = Dense(units=dense_units_for_extracting // 8, activation='relu')(dense_all_2)

    # Выходной слой
    output = Dense(units=out_layer, activation="sigmoid")(dense_all_3)

    # Создание модели сети Siamese
    siamese_model = Model(inputs=[*input_layers, vector_images_all_input, vector_main_image_input, bert_vector_input], outputs=output)

    # Создание слоев ввода для разветвления
    input_a = [*input_layers, vector_images_all_input, vector_main_image_input, bert_vector_input]
    input_b = []

    # Создание входных слоев для второй ветви
    for input_layer in input_a:
        new_input_layer = tf.keras.layers.Input(shape=input_layer.shape[1:], name=input_layer.name + "_b")
        input_b.append(new_input_layer)

    # Разветвление сети Siamese
    processed_a = siamese_model(input_a)
    processed_b = siamese_model(input_b)

    
    distance = Dense(units=1, activation="sigmoid")(tf.keras.layers.concatenate([processed_a, processed_b]))
    # Функции расстояния
    if func == "diff":
        # Функция "diff" вычисляет разность между векторами признаков образов A и B
        features_diff = tf.keras.layers.Subtract()([processed_a, processed_b])
        # Полносвязный слой для получения итогового значения расстояния
        distance = Dense(units=1, activation="sigmoid", name='distance')(features_diff)
    elif func == "cos_sim":
        # Функция "cos_sim" вычисляет косинусное сходство между векторами признаков образов A и B
        # Сначала выполняется нормализация векторов
        normalized_a = tf.keras.backend.l2_normalize(processed_a, axis=1)
        normalized_b = tf.keras.backend.l2_normalize(processed_b, axis=1)
        # Вычисление косинусного сходства
        cos_similarity = tf.keras.backend.dot(normalized_a, tf.keras.backend.transpose(normalized_b))
        # Приведение значения к диапазону [0, 1] с помощью линейного преобразования
        distance = Lambda(lambda x: (1 + x) / 2, name='distance')(cos_similarity)
    elif func == "prod":
        # Функция "prod" вычисляет произведение поэлементно между векторами признаков образов A и B
        features_prod = tf.keras.layers.Multiply()([processed_a, processed_b])
        # Полносвязный слой для получения итогового значения расстояния
        distance = Dense(units=1, activation="sigmoid", name='distance')(features_prod)
    elif func == "euclid":
        # Функция "euclid" вычисляет Евклидово расстояние между векторами признаков образов A и B
        features_diff = tf.keras.layers.Subtract()([processed_a, processed_b])
        euclidean_distance = tf.norm(features_diff, axis=1, keepdims=True)
        # Приведение значения к диапазону [0, 1] с помощью линейного преобразования
        distance = Lambda(lambda x: 1 / (1 + x), name='distance')(euclidean_distance)
    elif func == "concat":
        # Функция "concat" конкатенирует векторы признаков образов A и B
        concatenated = Concatenate(axis=-1)([processed_a, processed_b])
        # Инвертирование значений конкатенированного вектора
        negated_concatenated = Lambda(lambda x: -x)(concatenated)
        # Полносвязный слой для получения итогового значения расстояния
        distance = Dense(units=1, activation="sigmoid", name='distance')(negated_concatenated)



    # Полная модель сети Siamese
    siamese_network = Model(inputs=[*input_a, *input_b], outputs=distance)

    return siamese_network


# In[ ]:


def get_max_lists_count(column):
    """
    Функция для получения максимального количества списков в столбце.

    Аргументы:
    - column: столбец данных

    Возвращает:
    - max_lists_count: максимальное количество списков в столбце
    """
    max_lists_count = 0
    for row in column:
        if row is not None:
            max_lists_count = max(max_lists_count, len(row))
    return max_lists_count


def extend_lists_and_remove_images(column, max_lists):
    """
    Функция для расширения списков в столбце и удаления излишних изображений.

    Аргументы:
    - column: столбец данных
    - max_lists: максимальное количество списков

    Возвращает:
    - new_column: новый столбец данных с расширенными списками
    """
    new_column = []
    extend_arr = np.zeros((max_lists, 128), dtype=np.float32)
    for item in column:
        if item is None:
            item = extend_arr
        else:
            item = np.array([np.array(i, dtype=np.float32) for i in item], dtype=np.float32)
            item = item[:max_lists]
            extend_len = max_lists - len(item)
            if extend_len > 0:
                extend_arr[:extend_len] = 0
                item = np.concatenate([item, extend_arr[:extend_len]], axis=0)
        new_column.append(item)
    return new_column


def main_pic_column_converter(column):
    """
    Функция для конвертации столбца.

    Аргументы:
    - column: столбец данных

    Возвращает:
    - column: преобразованный столбец данных
    """
    column = column.fillna(pd.Series([np.zeros(128)]))
    column = column.apply(lambda x: [item for sublist in x for item in sublist])
    return column

def attributes_filter(column, min_value):
    """
    Функция для фильтрации атрибутов в столбце.

    Аргументы:
    - column: столбец данных
    - min_value: минимальное значение для отбора атрибутов

    Возвращает:
    - attributes_sorted: отфильтрованный и отсортированный список атрибутов
    """
    key_sorted = defaultdict(int)
    for line in column:
        if line is not None:
            for key in line:
                key_sorted[key] += 1
    key_filtered = {key: value for key, value in key_sorted.items() if value >= min_value}
    attributes_sorted = sorted(key_filtered.items(), key=lambda x: x[1], reverse=True)
    return attributes_sorted


def attributes_counter(key, column, mode, bound):
    """
    Функция для фильтрации и сортировки значений по ключу в столбце dataframe.

    Аргументы:
    - key: ключ для фильтрации и сортировки значений
    - column: столбец данных, содержащий словари
    - mode: режим вывода ('count' для значений, встречающихся больше bound раз, 'top' для топ bound самых часто встречаемых значений)
    - bound: количество значений в режиме

    Возвращает:
    - filtered_values: отфильтрованный и отсортированный список значений
    """

    key_counts = defaultdict(int)

    for line in column:
        if line is not None and key in line:
            value = line[key][0]  # Получаем значение по ключу из списка значений
            key_counts[value] += 1

    if mode == 'count':
        filtered_values = [value for value, count in key_counts.items() if count > bound]
    elif mode == 'top':
        filtered_values = heapq.nlargest(bound, key_counts, key=key_counts.get)
    else:
        raise ValueError("Недопустимый режим вывода. Допустимые значения: 'count' и 'top'.")

    return filtered_values


def merge_and_convert_in_tensors(df, pairs):
    """
    Функция для объединения и преобразования данных в тензоры.

    Аргументы:
    - df: исходный DataFrame
    - pairs: DataFrame с парами variantid

    Возвращает:
    - [tensors_1, tensors_2]: список тензоров для каждой пары данных
    """
    merged_df1 = pd.merge(pairs, df, left_on='variantid1', right_on='variantid', how='left')
    merged_df2 = pd.merge(pairs, df, left_on='variantid2', right_on='variantid', how='left')
    merged_df1 = merged_df1.drop(columns=["variantid1", "variantid2", "variantid"])
    merged_df2 = merged_df2.drop(columns=["variantid1", "variantid2", "variantid"])
    tensors_1 = []
    tensors_2 = []
    for column in merged_df1:   
        tensors_1.append(tf.convert_to_tensor(merged_df1[column].values.tolist()))
        tensors_2.append(tf.convert_to_tensor(merged_df2[column].values.tolist()))
    return [tensors_1, tensors_2]


# In[ ]:


# Чтение данных из файла Parquet в pandas DataFrame
df = pd.read_parquet('new_datasets/train_data.parquet')

# Определение лямбда-функции для загрузки JSON-строк
load = lambda x: loads(x) if x is not None else None

# Применение функции load для преобразования JSON-строк в объекты Python в двух столбцах DataFrame
df.characteristic_attributes_mapping = df.characteristic_attributes_mapping.apply(load)
df.categories = df.categories.apply(load)


# In[ ]:


# Нахождение максимального количества списков в столбце и сохранение в переменную maximum_in_all
maximum_list_count = get_max_lists_count(df["pic_embeddings_resnet_v1"])

# Расширение списков в столбце 'pic_embeddings_resnet_v1' до значения maximum_in_all и удаление лишних изображений
df["pic_embeddings_resnet_v1"] = extend_lists_and_remove_images(df["pic_embeddings_resnet_v1"], maximum_list_count)

# Понижение размерности столбца 'main_pic_embeddings_resnet_v1' и заполнение пустых ячеек
df["main_pic_embeddings_resnet_v1"] = main_pic_column_converter(column=df["main_pic_embeddings_resnet_v1"])


# In[ ]:


# Фильтрация столбца 'characteristic_attributes_mapping' на основе минимального значения и сортировка ключей
attributes_sorted = attributes_filter(column=df['characteristic_attributes_mapping'], min_value=200000)
key_sorted = [key for (key, value) in attributes_sorted]
least = [key for (key, value) in attributes_sorted if value <= 10000]
most = [key for (key, value) in attributes_sorted if value >= 10000]


# In[ ]:


# Создание нового DataFrame 'df_cats' с столбцом 'variantid'
df_cats = pd.DataFrame()
df_cats['variantid'] = df['variantid']
# Токенизация значений атрибутов для каждого категориального столбца
for cat_column in most:
    keys_cat = attributes_counter(cat_column, df['characteristic_attributes_mapping'], 'count', 1000)
    temp = []
    for entry in df['characteristic_attributes_mapping']:
        if entry == None:
            temp.append("rest")
        else:
            line = entry.get(cat_column)[0] if entry.get(cat_column) != None else "rest"
            if line in keys_cat:
                temp.append(line)
            else:
                temp.append("rest")
    df_cats[cat_column] = temp

for cat_column in least:
    keys_cat = attributes_counter(cat_column, df['characteristic_attributes_mapping'], 'top', 20)
    temp = []
    for entry in df['characteristic_attributes_mapping']:
        if entry is None:
            temp.append("rest")
        else:
            line = entry.get(cat_column)[0] if entry.get(cat_column) != None else "rest"
            if line in keys_cat:
                temp.append(line)
            else:
                temp.append("rest")
    df_cats[cat_column] = temp

# Токенизация значений категорий для столбца 'categories'
key_cats = defaultdict(int)
for line in df['categories']:
    if line is not None and '3' in line:
        value = line['3']
        key_cats[value] += 1
key_cats = [value for value, count in key_cats.items() if count > 1000]

temp = []
for entry in df['categories']:
    line = entry.get('3')
    if line in key_cats:
        temp.append(line)
    else:
        temp.append("rest")
df_cats['categories'] = temp


# In[ ]:


# Создание словаря 'cat_dicts' для хранения уникальных токенов для каждого категориального столбца
cat_dicts = {}

# Определение лямбда-функции для присвоения числовых идентификаторов токенам
is_rest = lambda x, n: 0 if n == 'rest' else (x+1)

# Обработка каждого столбца в 'df_cats'
for column in df_cats:
    if column == 'variantid':
        continue
    
    # Создание словаря категорий для текущего столбца
    categories = df_cats[column].unique()
    cat_dict = {cat: is_rest(i, str(cat)) for i, cat in enumerate(categories, start=0)}
    cat_dicts[column] = cat_dict
    
    # Замена значений в столбце на числовые идентификаторы
    df_cats[column] = df_cats[column].map(cat_dict)
df_cats


# In[ ]:


# Объединение DataFrame 'df_cats' и нужных столбцов из 'df' в новый DataFrame 'processed_df'
processed_df = df_cats.merge(df[["variantid", 'pic_embeddings_resnet_v1', 'main_pic_embeddings_resnet_v1', 'name_bert_64']], on='variantid')

# Чтение данных из файла Parquet в DataFrame 'df_pairs'
df_pairs = pd.read_parquet('new_datasets/train_pairs.parquet')

# Создание тензора 'Y' из столбца 'target' и удаление этого столбца из 'df_pairs'
Y = np.array(df_pairs.pop("target").values.tolist())

# Создание тензора 'X' путем объединения и преобразования данных из 'processed_df' и 'df_pairs'
X_lists = merge_and_convert_in_tensors(df=processed_df, pairs=df_pairs)


# In[ ]:





# In[ ]:





# In[ ]:


def pr_auc_macro(
    target_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    prec_level: float = 0.75,
    cat_column: str = "cat3_grouped"
) -> float:
    
    df = target_df.merge(predictions_df, on=["variantid1", "variantid2"])

    y_true = df["target"]
    y_pred = df["scores"]
    categories = df[cat_column]

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    # calculate metric for each big category
    for i, category in enumerate(unique_cats):
        # take just a certain category
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        # if there is no matches in the category then PRAUC=0
        if sum(y_true_cat) == 0:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # get coordinates (x, y) for (recall, precision) of PR-curve
        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        
        # reverse the lists so that x's are in ascending order (left to right)
        y = y[::-1]
        x = x[::-1]
        
        # get indices for x-coordinate (recall) where y-coordinate (precision) 
        # is higher than precision level (75% for our task)
        good_idx = np.where(y >= prec_level)[0]
        
        # if there are more than one such x's (at least one is always there, 
        # it's x=0 (recall=0)) we get a grid from x=0, to the rightest x 
        # with acceptable precision
        if len(good_idx) > 1:
            gt_prec_level_idx = np.arange(0, good_idx[-1] + 1)
        # if there is only one such x, then we have zeros in the top scores 
        # and the curve simply goes down sharply at x=0 and does not rise 
        # above the required precision: PRAUC=0
        else:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue
        
        # calculate category weight anyway
        weights.append(counts[i] / len(categories))
        # calculate PRAUC for all points where the rightest x 
        # still has required precision 
        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
        except ValueError:
            pr_aucs.append(0)
            
    return np.average(pr_aucs, weights=weights)


# In[ ]:





# In[ ]:


def objective(trial):
    num_filters_images = trial.suggest_int('num_filters_images', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 8)
    dense_units_for_extracting = trial.suggest_int('dense_units_for_extracting', 128, 1024)
    dense_units_for_extracting_main_image = trial.suggest_int('dense_units_for_extracting_main_image', 32, 512)
    embedding_dim = trial.suggest_int('embedding_dim', 8, 64)
    out_layer = trial.suggest_int('out_layer', 1, 16)
    dropout_coef = trial.suggest_float('dropout_coef', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    regular = trial.suggest_float('regular', 1e-4, 1e-1)
    func = trial.suggest_categorical('func', ['diff', 'cos_sim', 'prod', 'euclid', 'concat'])
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'softmax', 'elu', 'selu'])
    EPOCHS = trial.suggest_int('epochs', 10, 50)
    # Build the siamese network model
    siamese_model = build_siamese_network(cat_dicts, maximum_list_count, num_filters_images, kernel_size, dense_units_for_extracting, dense_units_for_extracting_main_image, embedding_dim, dropout_coef, activation, out_layer, func, regular)
    print()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy'

    siamese_model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.AUC(curve='PR')])
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    history = siamese_model.fit(X, Y, epochs=EPOCHS, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    val_auc_keys = [key for key in history.history.keys() if key.startswith('val_auc')]
    validation_scores = [history.history[key][-1] for key in val_auc_keys]
    validation_score = np.max(validation_scores)

    return validation_score


# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Access the best hyperparameters and results
best_params = study.best_params
best_value = study.best_value
print(best_params, best_value)


# In[ ]:


# Создание экземпляра модели Siamese Network
siamese_model = build_siamese_network(cat_dicts=cat_dicts, max_imgs_all=maximum_list_count, func="euclid", out_layer=64)

# Компиляция модели
siamese_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='PR')])
# Вывод сводки архитектуры модели
# Обучение модели

EPOCHS = 25
VALIDATION_SPLIT = 0.3
siamese_model.fit(X, Y, epochs=EPOCHS,validation_split=VALIDATION_SPLIT)


# In[ ]:


for layer in siamese_model.layers:
    layer_weights = layer.get_weights()  # Получение весов слоя
    print(layer_weights)


# In[ ]:


# Чтение данных из файла Parquet в pandas DataFrame
df_test = pd.read_parquet('new_datasets/test_data.parquet')

# Определение лямбда-функции для загрузки JSON-строк
load = lambda x: loads(x) if x is not None else None

# Применение функции load для преобразования JSON-строк в объекты Python в двух столбцах DataFrame
df_test.characteristic_attributes_mapping = df_test.characteristic_attributes_mapping.apply(load)
df_test.categories = df_test.categories.apply(load)


# In[ ]:


# Расширение списков в столбце 'pic_embeddings_resnet_v1' до значения maximum_in_all и удаление лишних изображений
df_test["pic_embeddings_resnet_v1"] = extend_lists_and_remove_images(df_test["pic_embeddings_resnet_v1"], maximum_list_count)

# Понижение размерности столбца 'main_pic_embeddings_resnet_v1' и заполнение пустых ячеек
df_test["main_pic_embeddings_resnet_v1"] = main_pic_column_converter(column=df_test["main_pic_embeddings_resnet_v1"])


# In[ ]:


# Создание нового DataFrame 'df_cats' с столбцом 'variantid'
df_test_cats = pd.DataFrame()
df_test_cats['variantid'] = df_test['variantid']

# Токенизация значений атрибутов для каждого категориального столбца
for cat_column in cat_dicts.keys():
    temp = []
    for entry in df_test['characteristic_attributes_mapping']:
        if entry is None:
            temp.append(0)
        else:
            line = entry.get(cat_column)
            if line is not None:
                line = ','.join(line)
                if line in cat_dicts[cat_column].keys():
                    temp.append(cat_dicts[cat_column][line])
                else:
                    temp.append(0)
            else:
                temp.append(0)
    df_test_cats[cat_column] = temp

# Токенизация значений категорий для столбца 'categories'
temp = []
for entry in df_test['categories']:
    line = entry.get('3')
    if entry is None:
        temp.append(0)
    else:
        if line in cat_dicts['categories'].keys():
            temp.append(cat_dicts[cat_column][line])
        else:
            temp.append(0)
df_test_cats['categories'] = temp


# In[ ]:


# Объединение DataFrame 'df_cats' и нужных столбцов из 'df' в новый DataFrame 'processed_df'
processed_df_test = df_test_cats.merge(df_test[["variantid", 'pic_embeddings_resnet_v1', 'main_pic_embeddings_resnet_v1', 'name_bert_64']], on='variantid')
# Чтение данных из файла Parquet в DataFrame 'df_test_pairs'
df_test_pairs = pd.read_parquet('new_datasets/test_pairs_wo_target.parquet')
test_out = df_test_pairs[["variantid1", "variantid2"]]
# Создание тензора 'X' путем объединения и преобразования данных из 'processed_df' и 'df_pairs'
X_test = merge_and_convert_in_tensors(df=processed_df_test, pairs=df_test_pairs)


# In[ ]:


predict = siamese_model.predict(X_test)
test_out["target"]= predict


# In[ ]:


test_out.to_csv('Answer2.csv', index=False)


# In[ ]:


test_out


# In[ ]:




