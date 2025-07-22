import sys
from src.database.connection import create_connection
from src.exception import CustomException

def insert_artwork(title, artist_name, style, prediction_result, image_path):
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            query = """
            INSERT INTO artuploads (title, artist_name, style, prediction_result, image_path)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (title, artist_name, style, prediction_result, image_path))
            conn.commit()
            cursor.close()
            conn.close()
    except Exception as e:
        raise CustomException(e, sys)
