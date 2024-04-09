import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def data_preparation(path):

    # Импортируем нужные данные
    df = pd.read_csv(path, encoding='utf-8', sep=',')

    # Удаляем столбец с целевой переменной
    df_data = df.drop('Target_regression', axis=1)

    # Список столбцов
    data_data_columns_name = ['Feature1', 'Feature2', 'Feature3', 'Feature4',
                              'Feature5']

    # выполним стандартизацию данных
    standart = StandardScaler()
    standart.fit(df_data)

    # Применяем трансформер
    standarted = standart.transform(df_data[data_data_columns_name])
    df_standard = pd.DataFrame(standarted, columns=data_data_columns_name)

    # выполним масштабирование данных
    scaler = MinMaxScaler()
    scaler.fit(df_standard)

    # Применяем трансформер
    scaled = scaler.transform(df_standard)
    df_scaled = pd.DataFrame(scaled, columns=data_data_columns_name)

    df_prepared = pd.concat([df_scaled, df['Target_regression']], axis=1)
    return df_prepared
