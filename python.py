from sklearn.decomposition import NMF
import numpy as np

# Create a sample data matrix with non-negative values
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Set the number of components for the factorization
n_components = 2

# Create an NMF object using the KL divergence algorithm
model = NMF(n_components=n_components, init='nndsvdar', solver='mu', beta_loss='kullback-leibler', max_iter=1000)

# Fit the model to the data matrix X
W = model.fit_transform(X)
H = model.components_

# Print the resulting factor matrices
print("Factor matrix W:")
print(W)
print("Factor matrix H:")
print(H)