# train_pipeline.py
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
from model import DeepNeuralNetwork 

def load_data_from_sql_pipeline():
    print("Extracting metadata from SQL tier via Pandas...")
    conn = sqlite3.connect("enterprise_data.db")
    
    query = "SELECT image_path, label FROM image_registry WHERE split_type = 'train'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    X_list = []
    Y_list = []
    
    print("Transforming and preprocessing image data streams into matrices...")
    for index, row in df.iterrows():
        img = Image.open(row['image_path']).convert('RGB')
        img = img.resize((64, 64))
        
        img_array = np.array(img)
        flattened_features = img_array.flatten()
        
        X_list.append(flattened_features)
        Y_list.append(row['label'])
        
    X = np.array(X_list).T
    Y = np.array(Y_list).reshape(1, len(Y_list))
    
    # Scale inputs within range [0, 1]
    X = X / 255.0
    
    return X, Y

def run_enterprise_system():
    print("Starting Complete Production AI & Data Pipeline System...")
    
    X_train, Y_train = load_data_from_sql_pipeline()
    num_features = X_train.shape[0]
    print(f"Extracted Array Dimensions -> Features: {num_features}, Training Examples: {X_train.shape[1]}")
    
    layers_dims = [num_features, 5, 3, 1] 
    
    print("Spawning DeepNeuralNetwork instance block...")
    model = DeepNeuralNetwork(
        layer_dims=layers_dims,
        learning_rate=0.0075,
        num_iterations=500  
    )
    
    print("Beginning mathematical execution fit optimization loop...")
    model.fit(X_train, Y_train, print_cost=True)
    
    print("Project System Complete and Fully Functional!")

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "init_db.py"])
    run_enterprise_system()