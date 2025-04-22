import json
import argparse
from datetime import datetime
from dine_sense import read_unique_stations, mark_stations_and_persons_on_video
from dotenv import load_dotenv
import os
import sys
import pymysql

# Load environment variables
if not load_dotenv(dotenv_path=".env"):
    print("❌ .env file not found or failed to load.")
    sys.exit(1)

# Access variables into a dictionary
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'cursorclass': pymysql.cursors.DictCursor
}

if not all([db_config['host'], db_config['user'], db_config['password'], db_config['database']]):
    print("❌ One or more DB environment variables are missing.")
    sys.exit(1)

def data_extraction(data=None):
    if isinstance(data, str):
        data = json.loads(data)

    try:
        frame_key = list(data.keys())[0]
        frame_count = int(frame_key.split('_')[1])
        timestamp = datetime.now()
        camera_id = "CAM_01"
        station_json = json.dumps(data[frame_key])

        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        query = """
            INSERT INTO Station_data (timestamp, frame_count, camera_id, station_json_data)
            VALUES (%s, %s, %s, %s)
        """
        values = (timestamp, frame_count, camera_id, station_json)

        cursor.execute(query, values)
        conn.commit()
        print("✅ Data inserted successfully.")

    except Exception as e:
        print(f"❌ Error inserting data into DB: {e}")
    
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

def insert_occupancy_data(data):
    if isinstance(data, str):
        data = json.loads(data)
    try:
        timestamp = datetime.now()
        camera_id = "CAM_01"
        frame_key = list(data.keys())[0]
        station_data = data[frame_key]

        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        query = """
            INSERT INTO Station_occupancy_stat (timestamp, camera_id, station_id, occupancy)
            VALUES (%s, %s, %s, %s)
        """

        for entry in station_data:
            values = (
                timestamp,
                camera_id,
                entry['station'],
                entry['occupants']
            )
            cursor.execute(query, values)

        conn.commit()
        print("✅ Occupancy data inserted successfully.")

    except Exception as e:
        print(f"❌ Error inserting occupancy data: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image and insert station data into the database.")
    parser.add_argument("image_path", help="Path to the input image file.")
    args = parser.parse_args()

    try:
        image_file_path = args.image_path
        csv_file_path = "SRC/calibration_file/updated_station_calibration.csv"
        output_video_path = "SRC/calibration_file/marked_video_with_persons_classified.mp4"

        unique_stations = read_unique_stations(csv_file_path)
        json_data = mark_stations_and_persons_on_video(image_file_path, unique_stations, output_video_path)

        data_extraction(json_data)
        insert_occupancy_data(json_data)

        print("✅ Image processing complete.")

    except Exception as e:
        print(f"❌ Main execution failed: {e}")

# Usage: python app/main.py output_frames/frame_03240.jpg

# Note: Ensure that the CSV file path and image file path are correct.