# Feature Engineering

In this project, we load data resulting from a simulatiom and calculate and store features using the raw data.

The input data has the following expected structure:

- "ClusterID": the cluster a new customer is assigned to, we use this to group customers in vehicle routes.
- "LocationID": the ID of the location.
- "Lat": the latitude of the customer location.
- "Lon": the longitude of the customer location.
- "ExpFillLevel": the expected fill level of the container (this can be considered the demand of a customer in a standard VRP)
- "Distance": the distance related to serving all customer in this cluster of customers.
- "ServiceLevel": the service level (how much of demand is fulfilled) in this cluster of customers.

You can of course modify to suit your own data structure.

In `main.py`, you can set various parameters related to the features. In `FeatureEngineer.py` the features are calculated, you can more if you want. The abstracted features are stored using a unique hash. In the machine learning module of this codebase you can load using this hash.