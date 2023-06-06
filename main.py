import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import random

rand = random.sample(range(1,8300), 83)

# Extract only 1/100 data
df = pd.DataFrame()
for cnt, chunk in enumerate(pd.read_csv('flight.csv', chunksize=10000)):
    print(f'{cnt}회 진행중')
    if (cnt in rand):
        df = pd.concat([df, chunk])

df.to_csv('sample.csv')


df = pd.read_csv('sample.csv')

# In case of transit, extract only the first airline
df['segmentsAirlineName'] = df['segmentsAirlineName'].apply(lambda x: x.split('||')[0] if '||' in x else x)

# In case of transit, combine duration
df['segmentsDurationInSeconds'] = df.apply(lambda row: sum(map(int, row['segmentsDurationInSeconds'].split('||'))) if pd.notnull(row['segmentsDurationInSeconds']) and '||' in row['segmentsDurationInSeconds'] else row['segmentsDurationInSeconds'], axis=1)

# If any class is not coach, delete that row and delete column 'segmentsCabinCode'
remove_idx = df[~df['segmentsCabinCode'].str.contains('coach|\\|\\|')].index
df.drop(remove_idx, inplace=True)

# Convert to datetime
df['searchDate'] = pd.to_datetime(df['searchDate'])
df['flightDate'] = pd.to_datetime(df['flightDate'])

# Calculate date difference and create new attribute 'dateDifference'
df['dateDifference'] = (df['flightDate'] - df['searchDate']).dt.days
df.drop(['searchDate', 'flightDate'], axis=1, inplace=True)

# Merge 'startingAirport' and 'estinationAirport' to 'route'
df['route'] = df['startingAirport'] + df['destinationAirport']
df.drop(['startingAirport', 'destinationAirport'], axis=1, inplace=True)

# Handling missing value
# Missing values of totalTravelDistance are replaced by the average of other totalTravelDistances with the same origin and destination
missing_rows = df[df['totalTravelDistance'].isna()].index
non_missing_rows = df[~df['totalTravelDistance'].isna()]
mean_distance = non_missing_rows.groupby(['route'])['totalTravelDistance'].mean()
df.loc[missing_rows, 'totalTravelDistance'] = df.loc[missing_rows].apply(lambda row: mean_distance.get((row['route'])), axis=1)

# Remove unnecessary attributes
df.drop(['legId',
         'fareBasisCode',
         'travelDuration',
         'baseFare',
         'isRefundable',
         'segmentsDepartureTimeEpochSeconds',
         'segmentsDepartureTimeRaw',
         'segmentsArrivalTimeEpochSeconds',
         'segmentsArrivalTimeRaw',
         'segmentsArrivalAirportCode',
         'segmentsDepartureAirportCode',
         'segmentsAirlineCode',
         'segmentsEquipmentDescription',
         'segmentsDistance',
         'segmentsCabinCode'], axis=1, inplace=True)



# Load data
data = df.copy()
data = pd.read_csv('test.csv')

# Seperate input and target data
X = data.drop('totalFare', axis=1)  # input
y = data['totalFare']  # target

# Split to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Encode categorical Variables usind OneHotEncoder
categorical_cols = ['route', 'isBasicEconomy', 'isNonStop', 'segmentsAirlineName']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(X_train[categorical_cols])
X_train_encoded = pd.DataFrame(encoder.transform(X_train[categorical_cols]))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]))

X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X_test_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Scale numeric variables using standardScaler
numeric_cols = ['elapsedDays', 'seatsRemaining', 'totalTravelDistance',
                'segmentsDurationInSeconds', 'dateDifference']
scaler = StandardScaler()
scaler.fit(X_train[numeric_cols])
X_train_scaled = pd.DataFrame(scaler.transform(X_train[numeric_cols]))
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numeric_cols]))

X_train_scaled.columns = numeric_cols
X_test_scaled.columns = numeric_cols

# Combine Encoded and Scaled data
X_train_preprocessed = pd.concat([X_train_encoded, X_train_scaled], axis=1)
X_test_preprocessed = pd.concat([X_test_encoded, X_test_scaled], axis=1)

# Create and Train Random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train_preprocessed, y_train)
rf_model.score(X_train_preprocessed, y_train)
rf_model.score(X_test_preprocessed, y_test)

# Predict
y_pred = rf_model.predict(X_test_preprocessed)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

plt.scatter(X_test['dateDifference'], y_pred)
plt.show()


# Compared to 60 days of actual data
# Substitute median values for 'totalTravelDistance' and 'segmentsDurationInSeconds'
length = 60
testset = {'elapsedDays': [0] * length,
           'isBasicEconomy': [False] * length,
           'isNonStop': [True] * length,
           'seatsRemaining': [9] * length,
           'totalTravelDistance': [X[X['route'] == 'DFWIAD']['totalTravelDistance'].median()] * length,
           'segmentsAirlineName': ['Delta'] * length,
           'segmentsDurationInSeconds': [X[X['route'] == 'DFWIAD']['segmentsDurationInSeconds'].median()] * length,
           'dateDifference': list(range(1, length + 1)),
           'route': ['DFWIAD'] * length
           }
testset = pd.DataFrame(testset)

# Encode and scale test set
testset_encoded = pd.DataFrame(encoder.transform(testset[categorical_cols]))
testset_encoded.columns = encoder.get_feature_names_out(categorical_cols)
testset_scaled = pd.DataFrame(scaler.transform(testset[numeric_cols]))
testset_scaled.columns = numeric_cols

testset_preprocessed = pd.concat([testset_encoded, testset_scaled], axis=1)

testy_pred = rf_model.predict(testset_preprocessed)

# 6/2 DFW -> IAD (Skyscanner)
DFWIAD_list = [
    334, 334, 334, 334, 334, 334, 323, 323, 323, 323, 323, 323,
    210, 212, 210, 212, 210, 212, 212, 212, 189, 189, 279, 201,
    189, 165, 189, 240, 210, 189, 201, 150, 130, 130, 130, 188,
    210, 188, 150, 150, 188, 189, 189, 188, 188, 130, 150, 201,
    201, 189, 188, 201, 130, 150, 189, 189, 189, 240, 189, 130]


plt.title('DFW -> IAD')
plt.plot(list(range(length)), testy_pred, label='model prediction')
plt.plot(list(range(length)), DFWIAD_list, label='skyscanner real data')
plt.xlabel('Departure date - Search date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

