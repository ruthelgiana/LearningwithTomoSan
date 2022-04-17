import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = pd.read_csv("01224099999.csv")
mse_data = pd.read_csv("01224099999.csv", usecols=[16,20])
 

x = data["WDSP"]
Y = data["MAX"]

# Error processing:reshape(-1,1)
X = np.array(x).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

scaler = StandardScaler()
clf = LassoCV(alphas=10 ** np.arange(-6, 1, 0.1), cv=5)

scaler.fit(x_train)
clf.fit(scaler.transform(x_train), y_train)

y_pred = clf.predict(scaler.transform(x_test))
mse = mean_squared_error(y_test, y_pred)
print(f"mse: {mse}")

# dt$station = station
# dt$station = as.numeric(format(dt$station, “%Y”))
# dt$station = as.numeric(format(dt$station, “%y”))

# dt$date = date
# dt$date = as.numeric(format(dt$date, “%m”))
# dt$date = format(dt$date, “%b”)
# dt$date = format(dt$date, “%B”)