def map_table_name(folder_name, table_name):
    if folder_name == "camera_7":
        mapping = {
            "002": "T7",
            "001": "T8",
            "003": "T9",
            "005": "T10",
            "004": "T11",
            "006": "T12",
            "009": "T18",
            "011": "T19",
            "008": "T21",
            "010": "T22",
            "007": "T15"         
        }
        return mapping.get(table_name)
    elif folder_name == "camera_3":
        mapping = {
            "001": "T22",
            "002": "T19",
            "003": "T20",
            "004": "T23",
            "005": "T21",
            "006": "T18"
        }
        return mapping.get(table_name)
    return table_name