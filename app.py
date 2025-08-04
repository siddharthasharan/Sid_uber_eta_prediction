import datetime
import pandas as pd
import streamlit as st
import holidays # importing the holidays library 
import sys, os
import re
from src import predict
from src import preprocessing
from PIL import Image
from math import radians, cos, sin, asin, sqrt, atan2, degrees

r = 6371  # Radius of earth in kilometers
direction_mapping = {'North': 0, 
                     'North-East': 45, 
                     'East': 90, 
                     'South-East': 135, 
                     'South': 180, 
                     'South-West': 225, 
                     'West': 270, 
                     'North-West': 315}

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate distance between two geolocation coordinates
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * r

def calculate_delivery_location(restaurant_latitude, restaurant_longitude, distance, direction):
    """
    Calculate the delivery location's latitude and longitude based on the distance and direction from the restaurant location
    """
    # Convert latitude and longitude to radians
    restaurant_latitude_radians = radians(float(restaurant_latitude))
    restaurant_longitude_radians = radians(float(restaurant_longitude))

    # Convert distance to radians
    distance_radians = distance / r
    # Convert direction to radians
    direction_radians = radians(direction_mapping[direction])

    # Calculate the delivery location's latitude and longitude
    delivery_latitude_radians = asin(sin(restaurant_latitude_radians) * cos(distance_radians) +
                                     cos(restaurant_latitude_radians) * sin(distance_radians) * cos(direction_radians))
    delivery_longitude_radians = restaurant_longitude_radians + atan2(sin(direction_radians) * sin(distance_radians) * cos(restaurant_latitude_radians),
                                                                       cos(distance_radians) - sin(restaurant_latitude_radians) * sin(delivery_latitude_radians))

    # Convert back to degrees
    delivery_latitude = degrees(delivery_latitude_radians)
    delivery_longitude = degrees(delivery_longitude_radians)

    return delivery_latitude, delivery_longitude


def validate_location_inputs(delivery_location_latitude, delivery_location_longitude,  restaurant_latitude, restaurant_longitude):
    """
       Validating the restaurant and delivery geolocations
       Validating the delivery location is within 20 km of the restaurant and within the Indian boundary

       if valid:
       Returns True

       else:
       prints a warning on app
       Returns False
    """
    valid_inputs = True

    distance = haversine(float(restaurant_longitude), float(restaurant_latitude), float(delivery_location_longitude), float(delivery_location_latitude))
    if distance > 20:
        st.warning(f"Delivery location is too far ({distance:.2f} km) from the restaurant. Maximum distance allowed is 20 km.")
        valid_inputs = False

        if float(delivery_location_latitude) < 6.76 or float(delivery_location_latitude) > 35.49:
            st.warning("The delivery location is outside India. Please enter a valid latitude value (between 6.76 and 35.49)")
            valid_inputs = False

        if float(delivery_location_longitude) < 68.11 or float(delivery_location_longitude) > 97.41:
            st.warning("The delivery location is outside India. Please enter a valid longitude value (between 68.11 and 97.41)")
            valid_inputs = False

    return valid_inputs


def get_user_input(df):

    # getting input for order
    st.sidebar.write(f"**Order Related Information**")
    date = st.sidebar.date_input("what is the Order Date?")
    st.sidebar.write("**Order_time**")
    order_time = datetime.datetime.now().strftime('%H:%M:%S')  # Get the current system time
    st.sidebar.write(f" :blue[{order_time}]")
    order_datetime = datetime.datetime.combine(date, datetime.datetime.strptime(order_time, '%H:%M:%S').time())
    st.sidebar.write("**Order_pickup_time**")
    pickup_time = order_datetime + datetime.timedelta(minutes=15)  # order_pickup_time is order_time + 15 mins
    st.sidebar.write(f" :blue[{str(pickup_time)}]")
    # st.sidebar.write("-"*8)
    order_type = st.sidebar.selectbox('What is the type of order?',
                                      df['Type_of_order'].unique())
    multiple_deliveries = st.sidebar.selectbox('How many deliveries are combined?',
                                               sorted(df['Multiple_deliveries'].unique().astype('int')))

    
    
    # Getting input for delivery geolocations
    st.sidebar.write(f"**Location Related Information**")

    # Extracting unique delivery_person_ids and creating a dictionary of restaurant locations
    # using raw_data.csv for delivery_person_ID as we dropped in preprocessing.py as it is not required for model predictions
    # here, we require for showing in frontend app
    
    raw_data = pd.read_csv("data/raw_data.csv")
    delivery_person_ids = raw_data["Delivery_person_ID"].unique()
    restaurant_locations = {}
    for delivery_person_id in delivery_person_ids:
        match = re.search(r'RES(\d+)DEL', delivery_person_id)
        if match:
            restaurant_number = int(match.group(1))
            city = delivery_person_id.split('RES')[0]
            row = raw_data.loc[raw_data["Delivery_person_ID"] == delivery_person_id]
            longitude = row["Restaurant_longitude"].values[0]
            latitude = row["Restaurant_latitude"].values[0]
            if city not in restaurant_locations:
                restaurant_locations[city] = {}
            restaurant_locations[city][restaurant_number] = (longitude, latitude)

    
    # Get user input for restaurant
    city_code = st.sidebar.selectbox('What is the city name of delivery?', sorted(restaurant_locations.keys()))
    restaurant_numbers = sorted(list(restaurant_locations[city_code].keys()))
    restaurant_number = st.sidebar.selectbox('Select the restaurant number', restaurant_numbers)
    restaurant_latitude, restaurant_longitude = restaurant_locations[city_code][restaurant_number]


    # Get user input for distance and direction from the restaurant
    distance = st.sidebar.selectbox('Select the distance (in km) from the restaurant', range(1, 21))
    direction = st.sidebar.selectbox('Select the direction from the restaurant', list(direction_mapping.keys()))


    # Calculate the delivery location's latitude and longitude
    delivery_location_latitude, delivery_location_longitude = calculate_delivery_location(restaurant_latitude, restaurant_longitude, distance, direction)
   
    valid_inputs = validate_location_inputs(delivery_location_latitude, delivery_location_longitude, restaurant_latitude, restaurant_longitude)

    if valid_inputs:      
        
        # Getting input for delivery person
        st.sidebar.write(f"**Delivery Person Related Information**")
        delivery_person_age = st.sidebar.slider("How old is the delivery person?",
                                                int(df['Delivery_person_Age'].min()),
                                                int(df['Delivery_person_Age'].max()),
                                                int(df['Delivery_person_Age'].mean()))
        
        delivery_person_rating = st.sidebar.slider("What is delivery person rating?",
                                                   float(df['Delivery_person_Ratings'].min()),
                                                   float(df['Delivery_person_Ratings'].max()),
                                                   float(df['Delivery_person_Ratings'].mean()))        
        
        
        # Getting input for vehicle type, condition of the delivery person
        vehicle = st.sidebar.selectbox('What type of vehicle delivery person has?',
                                       df['Type_of_vehicle'].unique())
        
        # Create a mapping dictionary for vehicle condition to do model predictions
        vehicle_condition_mapping = {
            'poor': 0,
            'not bad': 1,
            'good': 2,
            'excellent': 3
        }

        vehicle_condition_options = list(vehicle_condition_mapping.keys())

        vehicle_condition = st.sidebar.selectbox('What is the Vehicle condition of delivery person?',
                                                 vehicle_condition_options)
        
        
        # Getting input for the city type, which city of India
        st.sidebar.write(f"**City Related Information**")
        city_type = st.sidebar.selectbox(f'Which type of **city type** it is?',
                                    df['City_type'].unique())
        

        # Getting input for road, weather
        st.sidebar.write(f"**Weather Conditions/Event Related Information**")
        road_density = st.sidebar.selectbox('What is road traffic density?',
                                            df['Road_traffic_density'].unique())
        weather_conditions = st.sidebar.selectbox('How is the weather?',
                                                  df['Weather_conditions'].unique())
        

        # Getting input for validating if there is festival or not on selected date
        # Create a custom HolidayBase object and add specific Indian festivals for the current year
        current_year = date.year
        in_holidays = holidays.HolidayBase()
        in_holidays.append({f'26-01-{current_year}': 'Republic Day'})
        in_holidays.append({f'15-08-{current_year}': 'Independence Day'})
        in_holidays.append({f'02-10-{current_year}': 'Gandhi Jayanti'})
        in_holidays.append({f'25-12-{current_year}': 'Christmas Day'})
        in_holidays.append({f'01-01-{current_year}': 'New Year\'s Day'})
        in_holidays.append({f'14-04-{current_year}': 'Dr. Ambedkar Jayanti'})
        in_holidays.append({f'01-05-{current_year}': 'May Day'})
        in_holidays.append({f'19-08-{current_year}': 'Muharram'})
        in_holidays.append({f'02-09-{current_year}': 'Janmashtami'})
        in_holidays.append({f'08-10-{current_year}': 'Dussehra'})
        in_holidays.append({f'24-10-{current_year}': 'Diwali'})

        st.sidebar.write("Is there a festival?")
        if date in in_holidays:
            festival = "Yes"
            st.sidebar.write(f"**Festival today**")
        else:
            festival = "No"
            st.sidebar.write(f"**No festival**")

        submit_button = st.sidebar.button("Submit")        
        
        
        # the user input is put into a dataframe and then sent to model for predictions
        if submit_button:
            st.toast('Model is running! Prediction time updated!',icon='🏃')
            X = pd.DataFrame({
                'ID': '123456',
                'Delivery_person_ID': city_code + 'RES' + str(restaurant_number) + 'DEL02',
                'Delivery_person_Age': delivery_person_age,
                'Delivery_person_Ratings': delivery_person_rating,
                'Restaurant_latitude': restaurant_latitude,
                'Restaurant_longitude': restaurant_longitude,
                'Delivery_location_latitude': format(float(delivery_location_latitude), ".6f"),
                'Delivery_location_longitude': format(float(delivery_location_longitude), ".6f"),
                'Order_Date': date.strftime('%d-%m-%Y'),
                'Time_Ordered': order_time,
                'Time_Order_picked': pickup_time.strftime('%H:%M:%S'),
                'Weatherconditions': 'conditions ' + weather_conditions,
                'Road_traffic_density': road_density,
                'Vehicle_condition': vehicle_condition_mapping[vehicle_condition],
                'Type_of_order': order_type,
                'Type_of_vehicle': vehicle,
                'Multiple_deliveries': multiple_deliveries,
                'Festival': festival,
                'City_type': city_type
            }, index=[0])
            return X
    else:
        return None


if __name__ == "__main__":
    st.set_page_config(page_title="Food Delivery Time Prediction", 
                       page_icon=None, 
                       layout="centered",
                       initial_sidebar_state="auto")

    # Read in training data
    df = pd.read_csv("data/raw_data.csv")
    dataprocess = preprocessing.DataProcessing()
    dataprocess.cleaning_steps(df)

    # Displaying app header
    st.title("Food Delivery Time Prediction")

    # Displaying image of homepage 
    img = Image.open('./assets/food_delivery_README.jpg')
    st.image(img, width=700)

    st.write("""
                The food delivery time prediction model ensures prompt and accurate deliveries in the food industry.
                
                It leverages advanced data cleaning techniques, feature engineering, and considers order details, location, delivery person information, and weather conditions to provide accurate delivery time estimates.
             """)

    # create the sidebar
    st.sidebar.header("User Input Parameters")

    # create function for User input
    input_df = get_user_input(df)  # get user input from sidebar

    if input_df is not None:
        order_date = input_df['Order_Date'][0]
        order_time = input_df['Time_Ordered'][0]
        order_date_time = datetime.datetime.strptime(f'{order_date} {order_time}', '%d-%m-%Y %H:%M:%S')
        order_pickup_time = input_df['Time_Order_picked'][0]
        order_pickup_date_time = datetime.datetime.strptime(f'{order_date} {order_pickup_time}', '%d-%m-%Y %H:%M:%S')

        # get predictions
        # this is the output of the XGBRegressor
      
        total_delivery_minutes = round(predict.predict(input_df)[0], 2)  
        minutes = int(total_delivery_minutes)
        X = order_pickup_date_time + datetime.timedelta(minutes=minutes)

        # display predictions
        st.subheader(f"Order Details")
        st.write(f"**Order was Placed on :** :blue[{order_date_time}]")
        st.write(f"**Order was Picked up at :** :blue[{order_pickup_date_time}]")

        st.subheader(f"Prediction")
        formatted_X = "{:.2f}".format(total_delivery_minutes)
        st.write(f"**Total Delivery Time is :** :blue[{formatted_X} mins]")
        st.write(f"**Order will be delivered at approximately :** :blue[{X.strftime('%d-%m-%Y %H:%M')}]")