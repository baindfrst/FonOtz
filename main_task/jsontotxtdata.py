import json

def reviewsfile():
    with open("reviews_file.json", "r") as file:
        json.load(file)
