# init_db.py
import sqlite3
import os
from PIL import Image
import numpy as np

def initialize_production_database():
    print("Initializing Local Relational SQL Database...")
    db_name = "enterprise_data.db"
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create an image metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL UNIQUE,
            label INTEGER NOT NULL,
            split_type TEXT NOT NULL
        )
    """)
    conn.commit()
    
    # Create a mock 'images' directory and generate sample image files for testing
    os.makedirs("dataset/train", exist_ok=True)
    print("Generating test image assets on local disk...")
    
    sample_images = [
        ("dataset/train/cat_01.jpg", 1, "train"),
        ("dataset/train/dog_01.jpg", 0, "train"),
        ("dataset/train/cat_02.jpg", 1, "train"),
        ("dataset/train/dog_02.jpg", 0, "train"),
    ]
    
    for path, label, split in sample_images:
        if not os.path.exists(path):
            random_pixels = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(random_pixels)
            img.save(path)
            
        try:
            cursor.execute("""
                INSERT INTO image_registry (image_path, label, split_type)
                VALUES (?, ?, ?)
            """, (path, label, split))
        except sqlite3.IntegrityError:
            pass 
            
    conn.commit()
    
    cursor.execute("SELECT COUNT(*), label FROM image_registry GROUP BY label")
    metrics = cursor.fetchall()
    print(f"SQL Database populated successfully. Metrics (Count, Label): {metrics}")
    
    conn.close()

if __name__ == "__main__":
    initialize_production_database()