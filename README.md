## Before You Start

Before using the chatbot, please ensure that the `chat_model.h5` file exists in the project. This file contains the pre-trained model for the chatbot. If it doesn't exist, you will need to train the model by following these steps:

- Open the `model_training.ipynb` Jupyter notebook. 
- Run all the cells in the notebook to train the model.
- Once the training is complete, the `chat_model.h5` file will be generated.

## Usage

After confirming the presence of `chat_model.h5`, you can proceed to use the chatbot in your application.


## Run Locally

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirments.txt
```

Start Bot

```bash
  python main.py
```


## API Reference

#### Get all Data

```http
  GET https://databasetelegram.000webhostapp.com/fetchData.php
```



#### Get Data

```http
  GET https://databasetelegram.000webhostapp.com/getData.php?{user_id}
```

| Parameter | Type     |
| :-------- | :------- |
| `id` | `string` |


#### Insert Data

```http
  POST https://databasetelegram.000webhostapp.com/insertData.php
```

| Parameter | Type     |
| :-------- | :------- |
| `id` | `string` |
| `name` | `string` |
| `college` | `string` |
| `persona` | `string` |

#### Get all History

```http
  GET https://databasetelegram.000webhostapp.com/fetchHistory.php
```

#### Insert Data

```http
  POST https://databasetelegram.000webhostapp.com/insertHistory.php
```

| Parameter | Type     |
| :-------- | :------- |
| `id` | `string` |
| `message` | `string` |
| `answered` | `string` |


