# Sid_uber_eta_prediction
# Food Delivery Time Prediction using ML
<img src="assets/food_delivery_README.jpg" width="400">

## Project objective

**Background**
<font size="2">Assume Uber is a food delivery company that was launched in 2022. The users can select a restaurant to order any of the following food items: snacks, drinks, meals, buffet. The delivery partner normally uses a bicycle, electric scooter, scooter, or motorcycle to deliver the order.
</font>

**Goal**
<font size="2">Create an internal tool to estimate the time to deliver the food to the user, based on a set of given inputs. This will be used by other teams for enhancing driver experience, route optimization, capacity planning etc.
</font>

**Outputs**
<font size="2">Build a machine learning model to predict the time taken to deliver the food. Deploy the application using Streamlit Community Cloud with an easy to use UI, where the time to deliver is calculated based on some user inputs. Ensure that the code is clean, well organized, and document your findings (so your future self and other team members thank you!).
</font>

## Repository Structure

1. **assets**: Contains assets such as images of formulas, frontend etc.

2. **configs**: Centralized location for configuration files.

3. **data**: Stores different versions of data in distinct folders.
    - **data_after_feature_engineering.csv**: Dataset created after feature engineering
    - **data_cleaned.csv**: Dataset created after data cleaning
    - **raw_data.csv**: Raw dataset
    - **test.csv**: Testing dataset (can be optional)

4. **model**: Directory for saving and loading the model.pkl file.

5. **notebooks**: Google Colab notebooks for cleaning, preprocessing, feature engineering for reference

6. **references**: Contains documents with references used in the project.

7. **src**: Main source code directory with the following subfolders:
    - **preprocessing**: Functionality to preprocess, feature engineering, modelling on a raw dataset
    - **build_model**: Creates and saves the model.pkl file from the preprocessed dataset
    - **predict**: Predictions of saved model.pkl on new user input

8. **app.py**: Streamlit frontend

9. **Dockerfile**: Configuration for setting up the project in a Docker container.

## Data-dictionary

|Column|Description |
| :------------ |:---------------:|
|**ID**|order ID number| 
|**Delivery_person_ID**|ID number of the delivery partner|
|**Delivery_person_Age**|Age of the delivery partner|
|**Delivery_person_Ratings**|Ratings of the delivery partner based on past deliveries|
|**Restaurant_latitude**|The latitude of the restaurant|
|**Restaurant_longitude**|The longitude of the restaurant|
|**Delivery_location_latitude**|The latitude of the delivery location|
|**Delivery_location_longitude**|The longitude of the delivery location|
|**Order_Date**|Date of the order|
|**Time_Ordered**|Time the order was placed|
|**Time_Order_picked**|Time the order was picked|
|**Weatherconditions**|Weather conditions of the day|
|**Road_traffic_density**|Density of the traffic|
|**Vehicle_condition**|Condition of the vehicle|
|**Type_of_order**|The type of meal ordered by the customer|
|**Type_of_vehicle**|The type of vehicle delivery partner rides|
|**Multiple_deliveries**|Amount of deliveries driver picked|
|**Festival**|If there was a Festival or no.|
|**City_type**|Type of city, example metropolitan, semi-urban, urban.|
|**Time_taken(min)**| The time taken by the delivery partner to complete the order|

**Notebooks:**

1. [problem_statement](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/00_problem_statement.ipynb)

2. [complete_pipeline](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/05_complete_pipeline.ipynb)

3. [data_cleaning](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/01_data_cleaning.ipynb)

4. [data_eda](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/02_data_eda.ipynb)

5. [feature_engineering](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/03_feature_engineering.ipynb)

6. [preprocessing_modelling_feature_selection](https://github.com/PrepVector/applied-ml-uber-eta-prediction/blob/main/notebooks/04_preprocessing_modelling_feature_selection.ipynb)

These notebooks are a replica of the app version of the Uber ETA Prediction. The notebooks showcase data cleaning, exploratory data analysis, feature engineering, model selection in greater detail. 

## Setting Up the Project

### Prerequisites

- Docker installed on your machine.

### Instructions

1. Clone the repository:

    ```bash
    git clone  https://github.com/PrepVector/applied-ml-uber-eta-prediction.git
    ```

2. Build the Docker image:

    ```bash
    docker build -t fdt:latest .
    ```

3. Run the Docker container:

    ```bash
    docker run -it --rm --name "food_delivery_time" -p 8501:8501 fdt:latest
    ```

4. Access the Streamlit app in your web browser at [http://localhost:8501](http://localhost:8501).

### Additional Commands

- To enter the Docker container shell:

    ```bash
    docker run -it food_delivery_time /bin/bash
    ```

- To stop the running container:

    ```bash
    docker stop $(docker ps -q --filter ancestor=food_delivery_time)
    ```

Adjust the instructions based on your specific project needs.
