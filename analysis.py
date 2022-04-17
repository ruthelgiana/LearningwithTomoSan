import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.layers import LSTM, Dense
from keras.models import Sequential

import statsmodels.formula.api as smf

# source:National Oceanic and Atmospheric Administration, Department of Commerce
# url = https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/

class Analysis:

    def mse(object):
            
# Columns can be freely changed.
# Select two columns in the data(csv) file to load.
        x = data["WDSP"]
        Y = data["MAX"]

# Error processing:reshape(-1,1)
        X = np.array(x).reshape(-1,1)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        clf = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), cv=5)

        scaler.fit(x_train)
        clf.fit(scaler.transform(x_train), y_train)

        y_pred = clf.predict(scaler.transform(x_test))
        mse = mean_squared_error(y_test, y_pred)
        print(f"mse: {mse}")
        
    def LSTM(data):
        
        X = data["WDSP"]
        y = data["MAX"]

        y -= y.mean()

# Test size can be changed. Most are set at 0.3. Example: test_size = 0.1 or 0.2 or 0.3
        X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        
        x_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

# The data_dim,timesteps,num_classes,batch_size can also be changed numerically.
        data_dim = 40
        timesteps = 3
        num_classes = 10
        batch_size = 32
        
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, stateful=True,
                    batch_input_shape=(batch_size, timesteps, data_dim)))
        model.add(LSTM(32, return_sequences=True, stateful=True))
        model.add(LSTM(32, stateful=True))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        
# Generate pseudo-training data
# The batch_size can be changed.
        x_train = np.random.random((batch_size * 15000, timesteps, data_dim))
        y_train = np.random.random((batch_size * 15000, num_classes))

# Generate pseudo-verification data
# The batch_size can be changed.
        x_val = np.random.random((batch_size * 1500, timesteps, data_dim))
        y_val = np.random.random((batch_size * 1500, num_classes))

# The number of EPOCHS can also be changed. Example: epochs = 10
        model.fit(x_train, y_train,
              batch_size=batch_size, epochs=1, shuffle=False,
              validation_data=(x_val, y_val))
        
# You can see what kind of middle class we are in.  
        summary = model.summary()
        print(summary)
        
    def Summary(data):
# Each of the data and validation results can be seen at a glance.
        
        results = smf.ols("WDSP ~ MAX", data =data).fit()
        print(results.summary())
        
    
        
if __name__ == '__main__':
    data = pd.read_csv("01224099999.csv")
    
    # Analysis.mse(data)
    Analysis.LSTM(data)
    # Analysis.Summary(data)

# dt$station = station
# dt$station = as.numeric(format(dt$station, “%Y”))
# dt$station = as.numeric(format(dt$station, “%y”))

# dt$date = date
# dt$date = as.numeric(format(dt$date, “%m”))
# dt$date = format(dt$date, “%b”)
# dt$date = format(dt$date, “%B”)