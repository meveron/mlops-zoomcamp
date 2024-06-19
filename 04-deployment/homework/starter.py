import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def run():
    taxi_type = 'yellow'
    year = int(sys.argv[1])  # 2023
    month = int(sys.argv[2])  # 3
    output_dir = 'output'

    # Define file names and dirs
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'{output_dir}/{taxi_type}-{year:04d}-{month:02d}.parquet'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read data
    df = read_data(input_file)

    # Read model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Get predictions
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Standard deviation of prediction
    print(f'Std dev of predictions : {np.std(y_pred):.2f}')

    # Mean of prediction
    print(f'Mean of predictions : {np.mean(y_pred):.2f}')

    # Save result
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False)

if __name__ == '__main__':
    run()




