import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from PIL import Image
import glob

# Function to load and preprocess images
def load_images(image_paths):
    images = []
    for path in image_paths:
        print(f"Loading image: {path}")
        image = Image.open(path).convert('RGB')
        image = image.resize((100, 100))  # Resize to a manageable size
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        images.append(image.flatten())  # Flatten the image
    return np.array(images)

# Function to train the SOM
def train_som(data, x, y, iterations, learning_rate, sigma):
    print(f"Initializing SOM with learning_rate={learning_rate}, sigma={sigma}")
    som = MiniSom(x, y, data.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(data)
    print(f"Training SOM with {iterations} iterations")
    som.train_random(data, iterations)
    print("SOM training completed")
    return som

# Function to evaluate the SOM
def evaluate_som(som, data):
    print("Evaluating SOM...")
    labels = [som.winner(d) for d in data]
    labels = [x * som._weights.shape[1] + y for x, y in labels]  # Convert (x,y) to a single label

    unique_labels = len(np.unique(labels))
    print(f"Unique clusters: {unique_labels}")

    # Initialize metrics with -1
    silhouette = -1
    davies_bouldin = -1
    calinski_harabasz = -1

    if unique_labels < 2:
        print("Number of clusters is less than 2. Skipping silhouette, Davies-Bouldin, and Calinski-Harabasz score calculations.")
    else:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)

    return silhouette, davies_bouldin, calinski_harabasz

# Function to plot the SOM results
def plot_som_results(data, som, labels):
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(data):
        w = som.winner(x)
        plt.text(w[0] + 0.5, w[1] + 0.5, str(labels[i]), 
                 color=plt.cm.rainbow(labels[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
    plt.show()

# Load image data
print("Loading images...")
image_paths = glob.glob('C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/orginal.95.jpg')  # Replace with your image directory
data = load_images(image_paths)
print("Images loaded and preprocessed successfully.")

# Hyperparameter grid
learning_rates = [0.1, 0.5, 1.0]
sigmas = [0.5, 1.0, 1.5]
iterations = 10000  # Adjust based on your needs
x, y = 20, 20  # Increased SOM grid size

# Store results
results = []

print("Starting SOM training and evaluation...")
for lr in learning_rates:
    for sigma in sigmas:
        print(f"Training SOM with learning_rate={lr}, sigma={sigma}...")
        som = train_som(data, x, y, iterations, lr, sigma)
        silhouette, davies_bouldin, calinski_harabasz = evaluate_som(som, data)
        results.append((lr, sigma, silhouette, davies_bouldin, calinski_harabasz))
        print(f"Evaluation done: silhouette={silhouette}, davies_bouldin={davies_bouldin}, calinski_harabasz={calinski_harabasz}")

# Print results
print("All evaluations done. Results:")
for result in results:
    print(f"Learning Rate: {result[0]}, Sigma: {result[1]}, Silhouette Score: {result[2]}, Davies-Bouldin Index: {result[3]}, Calinski-Harabasz Index: {result[4]}")

# Find the best parameters
best_result = max(results, key=lambda item: (item[2], -item[3], item[4]))  # Prioritize silhouette, then davies_bouldin, then calinski_harabasz
best_lr, best_sigma = best_result[0], best_result[1]

print(f"Best parameters found: learning_rate={best_lr}, sigma={best_sigma}")

# Train and plot SOM with the best parameters
print("Training SOM with best parameters...")
best_som = train_som(data, x, y, iterations, best_lr, best_sigma)
labels = [best_som.winner(d) for d in data]
labels = [x * best_som._weights.shape[1] + y for x, y in labels]  # Convert (x,y) to a single label
print("Plotting SOM results...")
plot_som_results(data, best_som, labels)
print("Done.")
