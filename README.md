# Sid_uber_eta_prediction
This is a public repository for predicting ETA for Uber using ML
Project objective
Background Assume Uber is a food delivery company that was launched in 2022. The users can select a restaurant to order any of the following food items: snacks, drinks, meals, buffet. The delivery partner normally uses a bicycle, electric scooter, scooter, or motorcycle to deliver the order.

Goal Create an internal tool to estimate the time to deliver the food to the user, based on a set of given inputs. This will be used by other teams for enhancing driver experience, route optimization, capacity planning etc.

Outputs Build a machine learning model to predict the time taken to deliver the food. Deploy the application using Streamlit Community Cloud with an easy to use UI, where the time to deliver is calculated based on some user inputs. Ensure that the code is clean, well organized, and document your findings (so your future self and other team members thank you!).

Repository Structure
assets: Contains assets such as images of formulas, frontend etc.

configs: Centralized location for configuration files.

data: Stores different versions of data in distinct folders.

data_after_feature_engineering.csv: Dataset created after feature engineering
data_cleaned.csv: Dataset created after data cleaning
raw_data.csv: Raw dataset
test.csv: Testing dataset (can be optional)
model: Directory for saving and loading the model.pkl file.

notebooks: Google Colab notebooks for cleaning, preprocessing, feature engineering for reference

references: Contains documents with references used in the project.

src: Main source code directory with the following subfolders:

preprocessing: Functionality to preprocess, feature engineering, modelling on a raw dataset
build_model: Creates and saves the model.pkl file from the preprocessed dataset
predict: Predictions of saved model.pkl on new user input
app.py: Streamlit frontend

Dockerfile: Configuration for setting up the project in a Docker container.

Data-dictionary
Column	Description
ID	order ID number
Delivery_person_ID	ID number of the delivery partner
Delivery_person_Age	Age of the delivery partner
Delivery_person_Ratings	Ratings of the delivery partner based on past deliveries
Restaurant_latitude	The latitude of the restaurant
Restaurant_longitude	The longitude of the restaurant
Delivery_location_latitude	The latitude of the delivery location
Delivery_location_longitude	The longitude of the delivery location
Order_Date	Date of the order
Time_Ordered	Time the order was placed
Time_Order_picked	Time the order was picked
Weatherconditions	Weather conditions of the day
Road_traffic_density	Density of the traffic
Vehicle_condition	Condition of the vehicle
Type_of_order	The type of meal ordered by the customer
Type_of_vehicle	The type of vehicle delivery partner rides
Multiple_deliveries	Amount of deliveries driver picked
Festival	If there was a Festival or no.
City_type	Type of city, example metropolitan, semi-urban, urban.
Time_taken(min)	The time taken by the delivery partner to complete the order
