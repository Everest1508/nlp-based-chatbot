import requests
headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
def find_user_details(target_id):
    url = "https://databasetelegram.000webhostapp.com/getData.php"
    params = {'user_id': target_id} 
    response = requests.get(url, params=params,headers=headers)
    if response.status_code == 200:
        data = response.json()
        # print(data[0])
        return None if len(data)==0 else data[0]
    else:
        return None

def save_to_db(user_data):
    url = "https://databasetelegram.000webhostapp.com/insertData.php"
    data_to_insert = user_data
    response = requests.post(url, data=data_to_insert,headers=headers)
    if response.status_code == 200:
        print(response.text)
    else:
        print(f"Failed to insert data. Status code: {response.status_code}")

def save_history(user_data):
    url = "https://databasetelegram.000webhostapp.com/insertHistory.php"
    data_to_insert = user_data
    response = requests.post(url, data=data_to_insert,headers=headers)
    if response.status_code == 200:
        print(response.text)
    else:
        print(f"Failed to insert data. Status code: {response.status_code}")