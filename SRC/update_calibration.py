import csv
import os

def read_station_calibration(file_path):
    """
    Reads the station calibration CSV file and returns the data as a list of dictionaries.

    Args:
        file_path (str): Path to the station calibration CSV file.

    Returns:
        list: List of dictionaries containing station data.
    """
    stations = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            stations.append({
                "Station ID": row["Station ID"],
                "Table Position": row["Table Position"],
                "Chair Positions": row["Chair Positions"]
            })
    return stations

def find_unique_stations(stations):
    """
    Finds unique stations based on their Station ID.

    Args:
        stations (list): List of dictionaries containing station data.

    Returns:
        list: List of unique stations.
    """
    unique_stations = {}
    for station in stations:
        station_id = station["Station ID"]
        if station_id not in unique_stations:
            unique_stations[station_id] = station
    return list(unique_stations.values())

def save_unique_stations_to_csv(stations, output_path):
    """
    Saves the unique stations to a new CSV file.

    Args:
        stations (list): List of unique stations.
        output_path (str): Path to save the updated CSV file.
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Station ID", "Table Position", "Chair Positions"])
        for station in stations:
            writer.writerow([
                station["Station ID"],
                station["Table Position"],
                station["Chair Positions"]
            ])

if __name__ == "__main__":
    # Path to the input station calibration CSV file
    input_csv_path = "SRC/calibration_file/station_calibration.csv"

    # Path to save the updated station calibration CSV file
    output_csv_path = "SRC/calibration_file/updated_station_calibration.csv"

    # Read the station calibration data
    stations = read_station_calibration(input_csv_path)

    # Find unique stations based on Station ID
    unique_stations = find_unique_stations(stations)

    # Save the unique stations to a new CSV file
    save_unique_stations_to_csv(unique_stations, output_csv_path)

    print(f"Unique stations saved to: {output_csv_path}")