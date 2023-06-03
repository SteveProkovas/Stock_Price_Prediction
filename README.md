# Stock_Price_Prediction

    First, the necessary libraries are imported:
        tkinter: for creating the GUI.
        filedialog and messagebox from tkinter: for handling file selection and displaying error messages.
        pandas and numpy: for data manipulation and analysis.
        sklearn modules: for machine learning algorithms.
        statsmodels.tsa.arima_model: for the ARIMA algorithm.
        keras modules: for the LSTM algorithm.
        MinMaxScaler and mean_squared_error from sklearn.preprocessing: for data scaling and evaluating model performance.
        matplotlib.pyplot: for data visualization.

    The main window is created using tkinter and titled "Stock Price Predictor".

    The open_file function is defined to handle the file selection process. It opens a file dialog to select a CSV file and reads the data using pd.read_csv. If the file format is invalid, an error message is displayed.

    The show_data function displays the data in the window. It creates a data frame and data table to show the data preview. It also creates a label frame for selecting the machine learning algorithm. The algorithm options are displayed using the OptionMenu widget. Finally, a "Predict" button is created to trigger the prediction process.

    The predict function is responsible for making predictions based on the selected algorithm. It first preprocesses the data by dropping any missing values and standardizing the features using mean normalization. It then checks the selected algorithm and creates the corresponding model. For ARIMA and LSTM, additional steps are performed for data preparation and model training. Finally, the prediction result is displayed using a message box.

    The create_dataset function is used to prepare the dataset for the LSTM algorithm. It creates input sequences (data_X) and target values (data_Y) based on a given look-back value.

    The menu bar is created using tk.Menu and added to the main window. The "File" menu contains options for opening a file and exiting the program.

    The GUI main loop is started using window.mainloop(), which handles the event-driven programming and keeps the window open until the user closes it.
