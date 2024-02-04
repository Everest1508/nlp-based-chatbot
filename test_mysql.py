import requests

url = "https://databasetelegram.000webhostapp.com/getData.php"
params = {'user_id': 34253} 

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    # for row in data:
    print(data)
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
