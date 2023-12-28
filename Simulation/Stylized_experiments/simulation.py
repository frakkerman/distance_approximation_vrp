# %% Imports
import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
import time
import datetime

# %% Simulation Parameters
Horizondays = 200
InitialNrCustomers = 25
Iterations = 2
Replications = 1

# Demand and Arrival Parameters
LBdemand, UBdemand = 9, 11
LBarrivals, UBarrivals = 40, 60

# Vehicle Parameters
Vehicles = 5
VehicleCapacity = 80
DepotLoc = (50, 50)

# Location Bounds
MinX, MinY, MaxX, MaxY = 0, 0, 100, 100
MinXQuadrant, MaxXQuadrant = [0, 0, 50, 50], [50, 50, 100, 100]
MinYQuadrant, MaxYQuadrant = [0, 50, 50, 0], [50, 100, 100, 50]
ClusterChance = 0.7

# Save String
date_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
SaveString = f'ExperimentClusteredDaganzo_{Horizondays}days_{Iterations}iterations_{Replications}replications_{date_time}'

# %% Function Definitions

# Customer arrival clustered
def CustomerArrivals(NrOfCustomers, CustomerList, arrivalDay):
    rnd = random.randint(0, 3)
    for i in range(NrOfCustomers):
        if random.random() < ClusterChance:
            x = random.uniform(MinXQuadrant[rnd], MaxXQuadrant[rnd])
            y = random.uniform(MinYQuadrant[rnd], MaxYQuadrant[rnd])
        else:
            x = random.uniform(MinX, MaxX)
            y = random.uniform(MinY, MaxY)
        demand = random.randint(LBdemand, UBdemand)
        CustomerList.append(((x, y), demand, arrivalDay))

# Daganzo approximation (1984)
def DaganzoApproximation(PartialCustomerList):
    n = len(PartialCustomerList)
    Q = 25
    r = AvgDepotDist(CustomerList)
    A = 1000  # Area size
    return 2 * r * n / Q + 0.57 * math.sqrt(n * A)

# Euclidean distance
def distance(PointA, PointB):
    return math.sqrt((PointA[0] - PointB[0])**2 + (PointA[1] - PointB[1])**2)

# Avg distance to depot
def AvgDepotDist(CustomerList):
    return sum(distance(customer[0], DepotLoc) for customer in CustomerList) / len(CustomerList)

# Find Seeds for parallel VRP
def findSeeds(CustomerList):
    seed = []
    for s in range(Vehicles):
        furthestDistance = -1
        seedIndex = None
        for i, customer in enumerate(CustomerList):
            dist = distance(customer[0], DepotLoc)
            if s != 0:
                dist += sum(distance(customer[0], existing_seed[0]) for existing_seed in seed)
            if dist > furthestDistance:
                furthestDistance = dist
                seedIndex = i
        seed.append(CustomerList[seedIndex])
    return seed

# Parallel sweep seed assignment VRP heuristic
def parallelSeedAssignmentVRP(CustomerList):
    seeds = findSeeds(CustomerList)
    vehicleCustomerList = [[] for _ in range(Vehicles)]
    vehicleCapacityList = [VehicleCapacity] * Vehicles
    noFit = 0

    while noFit < len(seeds) * 5:
        for vehicle, seed in enumerate(seeds):
            cheapestInsert = float('inf')
            cheapestIndex = -1
            for i, customer in enumerate(CustomerList):
                if vehicleCapacityList[vehicle] - customer[1] >= 0:
                    dist = distance(customer[0], seed[0])
                    if dist < cheapestInsert:
                        cheapestInsert = dist
                        cheapestIndex = i
            if cheapestIndex >= 0:
                vehicleCustomerList[vehicle].append(CustomerList.pop(cheapestIndex))
                vehicleCapacityList[vehicle] -= vehicleCustomerList[vehicle][-1][1]
            else:
                noFit += 1

    return vehicleCustomerList, CustomerList

# Nearest neighbor
def nearestNeighbor(VehicleList):
    totalDist = 0
    vehicleRoutes = []

    for vehicle in VehicleList:
        routeDist = 0
        route = [DepotLoc]
        while vehicle:
            closest, closestDist = min(
                ((customer, distance(route[-1], customer[0])) for customer in vehicle),
                key=lambda x: x[1]
            )
            vehicle.remove(closest)
            route.append(closest[0])
            routeDist += closestDist
        routeDist += distance(route[-1], DepotLoc)
        route.append(DepotLoc)
        totalDist += routeDist
        vehicleRoutes.append(route)

    return totalDist, vehicleRoutes

# Feature calculation
def FeatureCalculation(routes, Featuresdf):
    # Each tuple in a route is (location, demand)
    # 'Featuresdf' is a Pandas DataFrame for storing the calculated features

    # Your code for calculating features goes here
    # Example (you will need to replace this with your actual feature calculation):
    for route in routes:
        # Calculate features for the route
        num_customers = len(route)
        total_demand = sum(customer[1] for customer in route)
        # Add more feature calculations as per your requirements

        # Append calculated features to the DataFrame
        new_row = {'NumCustomers': num_customers, 'TotalDemand': total_demand}  # Add other features here
        Featuresdf = Featuresdf.append(new_row, ignore_index=True)

    return Featuresdf

# Feature calculation for insertion
def FeatureCalculationinsertion(CustomerList):
    # Similar to FeatureCalculation, but specific to the context of insertion

    # Calculate features specific to insertion
    # Example (replace with features you like to test):
    num_customers = len(CustomerList)
    total_demand = sum(customer[1] for customer in CustomerList)
    # Add more insertion-specific feature calculations

    # Construct the feature array or DataFrame as per your model's requirements
    features = np.array([num_customers, total_demand])  # Add other features

    return features


def FindCheapestInsertions(CustomerList, costFunction, regressor):
    CheckCustomerList = []
    AcceptedCustomerList = []
    RemainingCapacity = Vehicles * VehicleCapacity
    NoNewCustomerAdded = False

    while not NoNewCustomerAdded:
        cheapestInsertionIndex = -1
        cheapestInsertion = float('inf')
        NoNewCustomerAdded = True

        for i, customer in enumerate(CustomerList):
            if RemainingCapacity - customer[1] > -1:
                NoNewCustomerAdded = False
                CheckCustomerList.append(customer)

                if costFunction == 'Daganzo':
                    insertionCosts = DaganzoApproximation(CheckCustomerList)
                elif costFunction == 'LinReg':
                    FeaturesSinglePred = FeatureCalculationinsertion(CheckCustomerList)
                    insertionCosts = regressor.predict_single(FeaturesSinglePred)

                CheckCustomerList.pop()
                if insertionCosts < cheapestInsertion:
                    cheapestInsertion = insertionCosts
                    cheapestInsertionIndex = i

        if cheapestInsertionIndex >= 0:
            CheckCustomerList.append(CustomerList[cheapestInsertionIndex])
            AcceptedCustomerList.append(CustomerList.pop(cheapestInsertionIndex))
            RemainingCapacity -= AcceptedCustomerList[-1][1]

    return AcceptedCustomerList


# Single LinReg prediction faster than SKLEARN
class BarebonesLinearRegression(LinearRegression):
    def predict_single(self, x):
        return np.dot(self.coef_, x) + self.intercept_

# Init simulation
def initIteration(CustomerList):   
    for i in range(InitialNrCustomers):
        x = random.uniform(MinX, MaxX)
        y = random.uniform(MinY, MaxY)
        demand = random.randint(LBdemand, UBdemand)
        CustomerList.append(((x, y), demand, -1))
    return CustomerList


# %% Simulation Execution
r2, Rmae, Rrmse, rejectedCustomers = [[] for _ in range(Replications)], [[] for _ in range(Replications)], [[] for _ in range(Replications)], [[] for _ in range(Replications)]
NrOfCustomers = [[] for _ in range(Replications)]
NrOfRemainingCustomers = [[] for _ in range(Replications)]
NrOfBeforeCustomers = [[] for _ in range(Replications)]
VRPDailyDistance = [[] for _ in range(Replications)]
start_time_sim = time.time()
bb_lin_reg = BarebonesLinearRegression()

for replication in range(Replications):
    random.seed(42 + replication)
    Featuresdf = pd.DataFrame()
    Targetdf = pd.DataFrame()
    regressor = None

    for iteration in range(Iterations):
        CustomerList = initIteration([])
        TotalVRPdistanceIter = 0

        for day in range(Horizondays):
            NrOfBeforeCustomers[replication].append(len(CustomerList))
            CustomerArrivals(random.randint(LBarrivals, UBarrivals), CustomerList, day)
            NrOfCustomers[replication].append(len(CustomerList))

            if iteration == 0:
                #in the first iteration we still use Daganzo for acceptance decisions
                AcceptedCustomerList = FindCheapestInsertions(CustomerList, 'Daganzo', regressor)
            else:
                AcceptedCustomerList = FindCheapestInsertions(CustomerList, 'LinReg', regressor)

            raise NotImplementedError("Provide your own VRP solver here.")
            vehicleRoute, VRPdistance, RejectedCustomers = None

            Featuresdf = FeatureCalculation(vehicleRoute, Featuresdf)
            Targetdf = Targetdf.append({'Target': VRPdistance}, ignore_index=True)

            CustomerList.extend(RejectedCustomers)
            rejectedCustomers[replication].append(len(RejectedCustomers))
            NrOfRemainingCustomers[replication].append(len(CustomerList))

        if len(Featuresdf) >= 200:
            x_train, x_test, y_train, y_test = train_test_split(Featuresdf, Targetdf, test_size=0.2, random_state=42+iteration)
        else:
            x_train, y_train = Featuresdf, Targetdf

        regressor = bb_lin_reg.fit(x_train, y_train)

        if len(Featuresdf) >= 200:
            y_pred = regressor.predict(x_test)
            r2[replication].append(metrics.r2_score(y_test, y_pred))
            Rmae[replication].append(metrics.mean_absolute_error(y_test, y_pred) / np.mean(y_test))
            Rrmse[replication].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)) / np.std(y_test))

        VRPDailyDistance[replication].append(TotalVRPdistanceIter / Horizondays)

print(f"[The simulation took {round((time.time() - start_time_sim) / 60)} minutes]")


# %% Storing Statistics
StorePerfStats = pd.DataFrame(columns=['IterationNo', 'ReplicationNo', 'R2', 'rMAE', 'rRMSE', 'AvgDailyVRPDistance'])