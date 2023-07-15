from datetime import timedelta
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.utils.dates import days_ago
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import pendulum

from SimpleNeuralNetwork import SimpleNeuralNetwork

def load_data(ti):
    X, y = make_multilabel_classification(n_samples=2020, n_features=10, n_classes=3, n_labels=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X[:2000], y[:2000], test_size=0.2, random_state=42)
    X_stream, y_stream = X[2000:], y[2000:]
    ti.xcom_push(key='train_data', value=(X_train, y_train))
    ti.xcom_push(key='test_data', value=(X_test, y_test))
    ti.xcom_push(key='stream_data', value=(X_stream, y_stream))

def train_model(ti):
    train_data = ti.xcom_pull(key='train_data', task_ids='load_data')
    X_train, y_train = train_data
    model = SimpleNeuralNetwork(learning_rate=0.01, epochs=5, activation='leaky_relu', loss_function='RMSE', layers=[7, 5, 6], classes=1)
    model.train(X_train, y_train)
    ti.xcom_push(key='model', value=model)

def make_predictions(ti):
    model = ti.xcom_pull(key='model', task_ids='train_model')
    stream_data = ti.xcom_pull(key='stream_data', task_ids='load_data')
    X_stream, y_stream = stream_data
    for X, y in zip(X_stream, y_stream):
        prediction = model.predict(X.reshape(1, -1))
        print(f"Predicted: {prediction}, Actual: {y}")

def print_predictions(ti):
    predictions = ti.xcom_pull(task_ids='make_predictions')
    print(predictions)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'provide_context': True,
}

dag = DAG(
    'simple_neural_network',
    default_args=default_args,
    description='A simple neural network using Airflow',
    schedule=timedelta(days=1),
    start_date=pendulum.today('UTC').add(days=-1),
    tags=['example'],
)

t1 = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t3 = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    dag=dag,
)

t4 = PythonOperator(
    task_id='print_predictions',
    python_callable=print_predictions,
)

t1 >> t2 >> t3 >> t4