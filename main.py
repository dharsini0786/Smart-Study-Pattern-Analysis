
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load dataset
df = pd.read_csv("students.csv")

print("STUDENT STUDY PATTERN ANALYSIS")
print("--------------------------------")

print("\nDataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Step 2: Select features
features = ["Study_Hours", "Attendance", "Sleep_Hours", "Marks"]

X = df[features]

# Step 3: Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = model.fit_predict(X_scaled)

# Step 5: Display results
print("\nCluster Results:")
print(df[["Student_ID", "Cluster"]])

print("\nCluster Summary:")
print(df.groupby("Cluster")[features].mean().round(2))

# Step 6: Save results
df.to_csv("clustered_students.csv", index=False)

print("\nResults saved to clustered_students.csv")

# Step 7: Create scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Study_Hours"],
    df["Marks"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Study Pattern Clustering")

plt.colorbar(label="Cluster")
plt.grid(True)

plt.savefig("study_pattern_graph.png")
plt.show()

# Step 8: Cluster comparison graph
cluster_means = df.groupby("Cluster")["Study_Hours"].mean()

plt.figure(figsize=(8, 5))

cluster_means.plot(kind="bar")

plt.xlabel("Cluster")
plt.ylabel("Average Study Hours")
plt.title("Average Study Hours in Each Cluster")

plt.savefig("cluster_comparison.png")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY!")
