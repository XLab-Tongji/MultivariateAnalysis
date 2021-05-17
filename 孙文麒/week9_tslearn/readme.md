# Week 9 TSLearn

## 1 Metrics

### 1.1 Dynamic Time Warping

**Base dtw_path method**

基础dtw方法调用

```python
path, sim = metrics.dtw_path(s_y1, s_y2)
```

**dtw_path method with custom metric**

支持不同类型的dtw计算

```python
# Example 1 : Length of the arc between two angles on a circle
def arc_length(angle_1, angle_2, r=1.):
    """Length of the arc between two angles (in rad) on a circle of
    radius r.
    """
    # Compute the angle between the two inputs between 0 and 2*pi.
    theta = np.mod(angle_2 - angle_1, 2*pi)
    if theta > pi:
        theta = theta - 2 * pi
    # Return the length of the arc
    L = r * np.abs(theta)
    return(L)


dataset_1 = random_walks(n_ts=n_ts, sz=sz, d=1)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=pi)  # Rescale the time series
dataset_scaled_1 = scaler.fit_transform(dataset_1)

# DTW using a function as the metric argument
path_1, sim_1 = metrics.dtw_path_from_metric(
    dataset_scaled_1[0], dataset_scaled_1[1], metric=arc_length
)

# Example 2 : Hamming distance between 2 multi-dimensional boolean time series
rw = random_walks(n_ts=n_ts, sz=sz, d=15, std=.3)
dataset_2 = np.mod(np.floor(rw), 4) == 0

# DTW using one of the options of sklearn.metrics.pairwise_distances
path_2, sim_2 = metrics.dtw_path_from_metric(
    dataset_2[0], dataset_2[1], metric="hamming"
)
```

## 2 Clustering and Barycenters

### 2.1 K-Shape

K-Shape聚类算法

```python
seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
# Keep first 3 classes and 50 first time series
X_train = X_train[y_train < 4]
X_train = X_train[:50]
numpy.random.shuffle(X_train)
# For this method to operate properly, prior scaling is required
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sz = X_train.shape[1]

# kShape clustering
ks = KShape(n_clusters=3, verbose=True, random_state=seed)
y_pred = ks.fit_predict(X_train)
```

### 2.2 K-means

K-means聚类算法，提供了内置的多种metric方案

```python
# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=3,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(X_train)

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)
```

## 3 Classification \(In brief\)

- SVM and GAK
- Learning Shapelets
- Early Classification
- Aligning discovered shapelets with timeseries

## 4 Nearest Neighbors \(In brief\)

- k-NN search