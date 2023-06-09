# flight-ticket-price-prediction

Dataset: https://www.kaggle.com/datasets/dilwong/flightprices

## Contributor
* 202135509 Kim daeun
* 201935014 Kim dohun
* 201935043 Mun heesang
* 201935138 Cha minjun

## Business Objective
> Observing the rise in flight bookings following the COVID-19 situation, we embarked on a project to predict to optimal time to book tickets from a traveler's perspective to secure the best price. The aim is determine how far in advance of the flight's departure a ticket should be booked to achieve the most cost-effective outcome in the post-pandemic travel landscape

## Data Exploration
|Column|Description|
|--|--|
|legId|An identifier for the flight|
|searchDate|The date (YYYY-MM-DD) of the flight|
|flightDate|The date (YYYY-MM-DD) of the flight|
|startingAirport|Three-character IATA airport code for the initial location|
|destinationAirport|Three-character IATA airport code for the arrival location|
|fareBasisCode|A code used by airlines to identify the type of fare|
|travelDuration|The travel duration in hours and minutes|
|elapsedDays|The number of elapsed days (usually 0)|
|isBasicEconomy|Boolean for whether the ticket is for basic economy|
|isRefundable|Boolean for whether the ticket is refundable|
|isNonStop|Boolean for whether the flight is non-stop|
|baseFare|The price of the ticket (in USD)|
|totalFare|The price of the ticket (in USD) including taxes and other fees|
|seatsRemaining|Integer for the number of seats remaining|
|totalTravelDistance|The total travel distance in miles. This data is sometimes missing|
|segmentsDepartureTimeEpochSeconds|String containing the departure time (Unix time) for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsDepartureTimeRaw|String containing the departure time (ISO 8601 format: YYYY-MM-DDThh:mm:ss.000±[hh]:00) for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsArrivalTimeEpochSeconds|String containing the arrival time (Unix time) for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsArrivalTimeRaw|String containing the arrival time (Unix time) for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsArrivalAirportCode|String containing the IATA airport code for the arrival location for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsDepartureAirportCode|String containing the IATA airport code for the departure location for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsAirlineName|String containing the name of the airline that services each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsAirlineCode|String containing the two-letter airline code that services each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsEquipmentDescription|String containing the type of airplane used for each leg of the trip (e.g. "Airbus A321" or "Boeing 737-800"). The entries for each of the legs are separated by '\|\|'|
|segmentsDurationInSeconds|String containing the duration of the flight (in seconds) for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsDistance|String containing the distance traveled (in miles) for each leg of the trip. The entries for each of the legs are separated by '\|\|'|
|segmentsCabinCode|String containing the cabin for each leg of the trip (e.g. "coach"). The entries for each of the legs are separated by '\|\|'|

## Data Preprocessing
* Cleaning dirty data<br>
    - The Nan value was processed as the mean value of the data having the same origin and destination
* Data reorganization<br>
    - In the case of (A Airline \|\| B Airline) of the ‘segmentsAirlineName’ property caused by transit, only the data of the departure point was left and the rest was deleted in order to change it to one data.
    - SegmentsDurationInSeconds property (A->passing time \|\| B->passing time) of the ‘segmentsDurationInSeconds’ property due to stopovers are added together and changed to total time
    - In order to use only those with coach (economy) rooms, including stopovers, a value other than coach or \|\| is detected in the property of ‘segmentsCabinCode’ and deleted.

* Feature engineering
    - Creates a new property representing the difference between searchDate and actual flight date by calculating the difference between searchDate and flightDate.
    - Combines startingAirport and destinationAirport into a property representing a single route.

* Remove attributes
    - Considered unnecessary while using various attributes in the model.

## Data Modeling and evaluation
### Linear regression
    Encoding - One Hot encoder
        For 'route', 'isBasicEconomy', 'isNonStop', 'segmentsAirlineName'
    
    Evaluation
        StandardSacler
            [Test size = 0.33] cv scores mean: 0.5130072431258934
            [Test size = 0.25] cv scores mean: 0.5121536385272675
            [Test size = 0.2] cv scores mean: 0.5124986601900334
        
        MinMaxScaler
            ✯ [Test size = 0.33] cv scores mean: 0.5125540629404329
            [Test size = 0.25] cv scores mean: 0.5120952482490219
            [Test size = 0.2] cv scores mean: 0.5115265354954621

        RobustScaler
            [Test size = 0.33] cv scores mean: 0.5130161001261946
            [Test size = 0.25] cv scores mean: 0.5121523899396516
            [Test size = 0.2] cv scores mean: 0.5125112252279671

        Normalizer
            [Test size = 0.33] cv scores mean: 0.49977525809928525
            [Test size = 0.25] cv scores mean: 0.49836537139402093
            [Test size = 0.2] cv scores mean: 0.4986612115562477
    
### Random Forest Regressor
    Encoding - One Hot encoder
        For 'route', 'isBasicEconomy', 'isNonStop', 'segmentsAirlineName'

    Scaling - StandardScaler

    Evaluation
        [N_estimators = 10]
            Train score: 0.9472398699525607
            Test score: 0.7529272018141125
            Mean Squared Error: 9149.822548262902
            Mean Absolute Error: 56.818976393567944

        [N_estimators = 30]
            Train score: 0.9559419653465212
            Test score: 0.7657378159254415
            Mean Squared Error: 8675.4083403307069
            Mean Absolute Error: 55.314613158927145

        [N_estimators = 100]
            Train score: 0.9589129877299448
            Test score: 0.770657133570329
            Mean Squared Error: 8493.231735541627
            Mean Absolute Error: 54.75890445842095

