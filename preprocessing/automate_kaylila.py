import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path, output_dir):

    df = pd.read_csv(input_path)

    df = df.drop('id', axis=1)

    categorical_cols = df.select_dtypes(include='object').columns

    imputer = SimpleImputer(strategy='most_frequent')

    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train).to_csv(f'{output_dir}/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{output_dir}/X_test.csv', index=False)

    pd.DataFrame(y_train).to_csv(f'{output_dir}/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{output_dir}/y_test.csv', index=False)

if __name__ == '__main__':
    preprocess_data(
        '../dataset_raw/healthcare-dataset-stroke-data.csv',
        'dataset_preprocessing'
    )