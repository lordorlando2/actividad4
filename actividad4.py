#importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Convertir el dataset a un data frame
data = pd.DataFrame({
        "origen": ["estacion_A", "estacion_B", "estacion_A", "estacion_E", "estacion_B", "estacion_D", "estacion_F", "estacion_E", "estacion_F", "estacion_A", "estacion_D", "estacion_B", "estacion_D"],
            "destino": ["estacion_E", "estacion_F", "estacion_D", "estacion_F", "estacion_D", "estacion_D", "estacion_F", "estacion_D", "estacion_D", "estacion_E", "estacion_D", "estacion_D", "estacion_D"]
            })

# Encode the origen and destino columns
le = LabelEncoder()
data["origen_encoded"] = le.fit_transform(data["origen"])
data["destino_encoded"] = le.fit_transform(data["destino"])

# Create the KMeans model
kmeans = KMeans(n_clusters=3, n_init=10)

# Fit the model
kmeans.fit(data[["origen_encoded", "destino_encoded"]])

# Get the cluster labels
labels = kmeans.predict(data[["origen_encoded", "destino_encoded"]])

# Visualize the results
plt.scatter(data["origen_encoded"], data["destino_encoded"], c=labels)
plt.show()

# Print the encoded data
print(data[["origen_encoded", "destino_encoded"]])

# Print the cluster labels
print(kmeans.labels_)
