import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("house_rent.csv")
#print(df.head())

#2)
#print(df.shape)
#print(df.info)
#print(df.describe())

#3)
#print(df.isnull())
df["age"].fillna(df["age"].mean(), inplace=True)
#print(df)
#print(df.duplicated())

#4)
x = df[["size_sqft","bedrooms","age"]]
y = df[["rent"]]

#5)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#6)
model = LinearRegression()
model.fit(x_train,y_train)

#7)
y_pred = model.predict(x_test)
#print(y_pred)

#8)
mae = mean_absolute_error(y_test, y_pred)
print(mae)

#10)The model shows that house size (size_sqft) has the strongest influence on rent, followed by the number of bedrooms, while age of the house has a smaller negative impact.
# Overall, the model’s performance is acceptable, as the prediction error is relatively low compared to the average rent, though the small dataset size limits reliability.

