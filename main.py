import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Create main window
window = tk.Tk()
window.title('Stock Price Predictor')

# Function to handle file selection
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
        show_data(data)
    else:
        messagebox.showerror('Error', 'Invalid file format. Please select a CSV file.')

# Function to display data in window
def show_data(data):
    data_frame = tk.Frame(window)
    data_frame.pack()

    # Create data table
    data_table = tk.LabelFrame(data_frame, text='Data')
    data_table.pack(side=tk.LEFT, padx=10, pady=10)

    tk.Label(data_table, text='Data Preview').grid(row=0, column=0)
    data_preview = tk.Label(data_table, text=data.head())
    data_preview.grid(row=1, column=0)

    # Create options for machine learning algorithm
    ml_options = tk.LabelFrame(data_frame, text='Machine Learning Algorithm')
    ml_options.pack(side=tk.LEFT, padx=10, pady=10)

    algorithm_var = tk.StringVar()
    algorithm_var.set('Linear Regression')

    tk.Label(ml_options, text='Select Algorithm').grid(row=0, column=0)
    algorithm_menu = tk.OptionMenu(ml_options, algorithm_var, 'Linear Regression', 'Decision Tree Regression',
                                  'Random Forest Regression', 'ARIMA', 'LSTM')
    algorithm_menu.grid(row=1, column=0)

    # Create button for making predictions
    predict_button = tk.Button(data_frame, text='Predict', command=lambda: predict(data, algorithm_var.get()))
    predict_button.pack(side=tk.LEFT, padx=10, pady=10)

# Function to make predictions and display results
def predict(data, algorithm):
    data.dropna(inplace=True)
    X = data[['Volume', 'Dividends', 'EPS', 'P/E Ratio']]
    y = data['Closing Price']
    X = (X - X.mean()) / X.std()

    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Decision Tree Regression':
        model = DecisionTreeRegressor()
    elif algorithm == 'Random Forest Regression':
        model = RandomForestRegressor()
    elif algorithm == 'ARIMA':
        model = ARIMA(y, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        y_pred = model_fit.forecast()[0]
    elif algorithm == 'LSTM':
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(y.values.reshape(-1, 1))
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        X_train, y_train = create_dataset(train, look_back=1)
        X_test, y_test = create_dataset(test, look_back=1)
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        y_pred = scaler.inverse_transform(test_predict)[-1][0]

    # Display prediction result
    messagebox.showinfo('Prediction', f'The predicted closing price is: {y_pred:.2f}')

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    data_X, data_Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        data_X.append(a)
        data_Y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_Y)

# Create menu and file functions
menu = tk.Menu(window)
window.config(menu=menu)

file_menu = tk.Menu(menu)
menu.add_cascade(label='File', menu=file_menu)
file_menu.add_command(label='Open', command=open_file)
file_menu.add_command(label='Exit', command=window.quit)

# Start the GUI main loop
window.mainloop()

