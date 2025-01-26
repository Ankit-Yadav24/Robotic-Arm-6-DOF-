import streamlit as st
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import os

# File paths for storing data
PRODUCTS_FILE = "products.xlsx"
ORDERS_FILE = "orders.xlsx"
OUT_OF_STOCK_FILE = "out_of_stock.xlsx"

# Ensure data files exist
for file in [PRODUCTS_FILE, ORDERS_FILE, OUT_OF_STOCK_FILE]:
    if not os.path.exists(file):
        pd.DataFrame().to_excel(file, index=False)

# Initialize YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO("grocery_model.h5", task="detect")
    except Exception as e:
        return st.error(f"Error loading model: {str(e)}")

model = load_model()

# Functions for data management
def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except:
        return pd.DataFrame()

def save_data(df, file_path):
    df.to_excel(file_path, index=False)

def authenticate_shopkeeper(username, password):
    # Dummy credentials
    return username == "shopkeeper" and password == "admin123"

# Shopkeeper Dashboard
def shopkeeper_dashboard():
    st.title("Shopkeeper Dashboard")

    # Tabs for different functionalities
    tabs = st.tabs(["Add Product", "View Products", "Out of Stock"])

    # Add Product Tab
    with tabs[0]:
        st.subheader("Add New Product")
        name = st.text_input("Product Name")
        category = st.text_input("Category")
        price = st.number_input("Price", min_value=0.0, step=0.01)
        quantity = st.number_input("Quantity", min_value=0, step=1)
        min_stock = st.number_input("Minimum Stock", min_value=0, step=1, value=10)
        object_class = st.text_input("Object Class (YOLO)")
        location = st.text_input("Location")
        description = st.text_area("Description")
        image = st.file_uploader("Product Image", type=["jpg", "png", "jpeg"])

        if st.button("Add Product"):
            if name and category and price > 0:
                products = load_data(PRODUCTS_FILE)
                new_product = {
                    "Name": name,
                    "Category": category,
                    "Price": price,
                    "Quantity": quantity,
                    "Minimum Stock": min_stock,
                    "Object Class": object_class,
                    "Location": location,
                    "Description": description,
                    "Last Updated": datetime.now(),
                }
                if image:
                    new_product["Image"] = image.getvalue()

                products = pd.concat([products, pd.DataFrame([new_product])], ignore_index=True)
                save_data(products, PRODUCTS_FILE)
                st.success("Product added successfully!")
            else:
                st.error("Please fill all fields correctly!")

    # View Products Tab
    with tabs[1]:
        st.subheader("View and Edit Products")
        products = load_data(PRODUCTS_FILE)
        if not products.empty:
            edited_products = st.experimental_data_editor(products)
            save_data(edited_products, PRODUCTS_FILE)
            st.success("Product data updated!")
        else:
            st.warning("No products available!")

    # Out of Stock Tab
    with tabs[2]:
        st.subheader("Out of Stock Products")
        products = load_data(PRODUCTS_FILE)
        if not products.empty:
            out_of_stock = products[products["Quantity"] <= 0]
            if not out_of_stock.empty:
                st.dataframe(out_of_stock)
            else:
                st.info("No products are out of stock!")
        else:
            st.warning("No products available!")

# Customer Interface
def customer_interface():
    st.title("Customer Interface")

    products = load_data(PRODUCTS_FILE)
    if products.empty:
        st.warning("No products available!")
        return

    st.subheader("Select Products")
    selected_products = []
    total_amount = 0

    for i, row in products.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{row['Name']}** - ${row['Price']:.2f}")
        with col2:
            qty = st.number_input(f"Quantity for {row['Name']}", min_value=0, max_value=row["Quantity"], step=1, key=row["Name"])
            if qty > 0:
                selected_products.append({"Name": row["Name"], "Price": row["Price"], "Quantity": qty})
                total_amount += qty * row["Price"]

    st.write(f"**Total Amount:** ${total_amount:.2f}")

    if st.button("Submit Order"):
        orders = load_data(ORDERS_FILE)
        new_order = pd.DataFrame(selected_products)
        new_order["Order Date"] = datetime.now()
        orders = pd.concat([orders, new_order], ignore_index=True)
        save_data(orders, ORDERS_FILE)
        st.success("Order submitted! Starting object detection...")

        # Object Detection
        if "ESP32_IP" not in st.session_state:
            st.session_state.ESP32_IP = st.text_input("Enter ESP32 Camera IP (e.g., http://192.168.0.101:81/stream)", key="camera_ip")

        if st.button("Start Detection", key="start_detection"):
            if st.session_state.ESP32_IP:
                cap = cv2.VideoCapture(st.session_state.ESP32_IP)
                if cap.isOpened():
                    st.text("Object detection in progress...")
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from ESP32 camera!")
                            break

                        results = model(frame)
                        for box in results[0].boxes:
                            class_name = results.names[int(box.cls)]
                            x1, y1, x2, y2 = map(int, box.xyxy)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        st.image(frame, channels="BGR")

                else:
                    st.error("Failed to connect to ESP32 camera!")

# Main App
def main():
    st.sidebar.title("Navigation")
    options = ["Customer Interface", "Shopkeeper Login"]
    choice = st.sidebar.radio("Choose an option:", options)

    if choice == "Shopkeeper Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_shopkeeper(username, password):
                shopkeeper_dashboard()
            else:
                st.error("Invalid credentials!")
    else:
        customer_interface()

if __name__ == "__main__":
    main()
