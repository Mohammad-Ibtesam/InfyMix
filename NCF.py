import numpy as np
import pandas as pd
import torch.nn as nn
import pickle
import os



class NCF_Layer():
  def __init__(self, number_of_neurons, input_length, learning_rate, is_last_layer=False):
    # Initialization for weights
    self.weights = np.random.randn(input_length, number_of_neurons) * np.sqrt(2.0 / input_length)
    self.bias = np.zeros((1, number_of_neurons))

    self.learning_rate = learning_rate
    self.is_last_layer = is_last_layer
    self.X_input = None
    self.Z = None
    self.A = None

  def forward(self, X):
    self.X_input = X
    self.Z = np.dot(self.X_input, self.weights) + self.bias

    if self.is_last_layer:
      self.A = np.ones_like(self.Z) # Derivative for linear activation is 1
      return self.Z 
    else:
      self.A = (self.Z > 0).astype(float) # Derivative for ReLU
      return np.maximum(self.Z, 0)

  def backward(self, dL_dA_current):
    dL_dZ = dL_dA_current * self.A
    dL_dW = np.dot(self.X_input.T, dL_dZ)
    dL_dB = np.sum(dL_dZ, axis=0, keepdims=True)
    dL_dX_input = np.dot(dL_dZ, self.weights.T)

    self.weights -= self.learning_rate * dL_dW
    self.bias -= self.learning_rate * dL_dB

    return dL_dX_input


class NCF_GMF_Component():
  def __init__(self, total_users, total_items, embedding_dim_gmf, learning_rate):
    self.embedding_dim_gmf = embedding_dim_gmf

    self.user_embedding_gmf = nn.Embedding(
    total_users, embedding_dim_gmf).weight.detach().numpy()

    self.item_embedding_gmf = nn.Embedding(
    total_items, embedding_dim_gmf).weight.detach().numpy()

    self.learning_rate = learning_rate
    self.user_idx = None
    self.item_idx = None

  def forward(self, user_idx, item_idx):

    self.user_idx, self.item_idx = user_idx, item_idx

    element_wise_product =  (self.user_embedding_gmf[user_idx]
            * self.item_embedding_gmf[item_idx])
    return element_wise_product

  def backward(self, gradient_from_ncf):
    dL_du_gmf = gradient_from_ncf * self.item_embedding_gmf[self.item_idx]
    dL_dv_gmf = gradient_from_ncf * self.user_embedding_gmf[self.user_idx]

    self.user_embedding_gmf[self.user_idx] -= self.learning_rate * dL_du_gmf
    self.item_embedding_gmf[self.item_idx] -= self.learning_rate * dL_dv_gmf


class NCF_MLP_Component():
  def __init__(self, total_users, total_items, embedding_dim_mlp, neurons_per_layer, learning_rate):
    self.embedding_dim_mlp = embedding_dim_mlp
    self.learning_rate = learning_rate

    self.user_embedding_mlp = nn.Embedding(
    total_users, embedding_dim_mlp).weight.detach().numpy()

    self.item_embedding_mlp = nn.Embedding(
    total_items, embedding_dim_mlp).weight.detach().numpy()

    self.layers = []
    input_length = 2 * embedding_dim_mlp # User embedding concatenated with item embedding

    for i, num_neurons in enumerate(neurons_per_layer):
      is_last_layer = (i == len(neurons_per_layer) - 1)
      self.layers.append(NCF_Layer(num_neurons, input_length, learning_rate, is_last_layer=is_last_layer))
      input_length = num_neurons

    self.user_idx = None
    self.item_idx = None
    self.combined_embedding = None

  def forward(self, user_idx, item_idx):
    self.user_idx = user_idx
    self.item_idx = item_idx

    user_emb = self.user_embedding_mlp[user_idx]
    item_emb = self.item_embedding_mlp[item_idx]

    self.combined_embedding = np.hstack([user_emb, item_emb]).reshape(1, -1)

    x = self.combined_embedding
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, gradient_from_ncf):

    gradient = gradient_from_ncf
    for layer in reversed(self.layers):
      gradient = layer.backward(gradient)

    dL_d_combined_embedding = gradient.flatten()
    dL_du_mlp = dL_d_combined_embedding[:self.embedding_dim_mlp]
    dL_dv_mlp = dL_d_combined_embedding[self.embedding_dim_mlp:]

    self.user_embedding_mlp[self.user_idx] -= self.learning_rate * dL_du_mlp
    self.item_embedding_mlp[self.item_idx] -= self.learning_rate * dL_dv_mlp


class NCF():
  def __init__(self, total_users, total_items, mlp_neurons_per_layer, learning_rate = 0.01, embedding_dim_gmf = 32, embedding_dim_mlp = 64):
    self.filename = 'NCF_model.pkl'
    self.learning_rate = learning_rate
    self.gmf_component = NCF_GMF_Component(total_users, total_items, embedding_dim_gmf, learning_rate)
    self.mlp_component = NCF_MLP_Component(total_users, total_items, embedding_dim_mlp, mlp_neurons_per_layer, learning_rate)

    input_dim_prediction_layer = embedding_dim_gmf + mlp_neurons_per_layer[-1]

    self.h = np.random.randn(input_dim_prediction_layer, 1) * np.sqrt(1 / input_dim_prediction_layer)
    self.b = np.zeros((1, 1))

    self.combined_output_gmf_mlp = None
    self.recommendations = []

  def forward(self, user_idx, item_idx):
    gmf_output = self.gmf_component.forward(user_idx, item_idx).reshape(1, -1)
    mlp_output = self.mlp_component.forward(user_idx, item_idx)

    self.combined_output_gmf_mlp = np.hstack([gmf_output, mlp_output])

    prediction = 1 / (1 + np.exp(-(np.dot(self.combined_output_gmf_mlp, self.h) + self.b)))
    return prediction

  def backward(self, gradient_from_loss):
    dL_d_sigmoid = gradient_from_loss
    sigmoid_output = 1 / (1 + np.exp(-(np.dot(self.combined_output_gmf_mlp, self.h) + self.b)))
    dL_d_prediction = dL_d_sigmoid * sigmoid_output * (1 - sigmoid_output)

    dL_d_h = np.dot(self.combined_output_gmf_mlp.T, dL_d_prediction)
    dL_d_b = np.sum(dL_d_prediction, axis=0, keepdims=True)

    self.h -= self.learning_rate * dL_d_h
    self.b -= self.learning_rate * dL_d_b

    dL_d_combined_output = np.dot(dL_d_prediction, self.h.T)

    gradient_to_gmf = dL_d_combined_output[:, :self.gmf_component.embedding_dim_gmf]
    gradient_to_mlp = dL_d_combined_output[:, self.gmf_component.embedding_dim_gmf:]

    self.gmf_component.backward(gradient_to_gmf.flatten())
    self.mlp_component.backward(gradient_to_mlp)

  def train(self, interacted_pairs, y_true, epochs = 50):

    interacted_pairs = np.array(interacted_pairs)
    y_true = y_true.reshape(-1, 1)

    for epoch in range(epochs):
      epoch_loss = 0

      permutation = np.random.permutation(len(interacted_pairs))
      shuffled_interacted_pairs = interacted_pairs[permutation]
      shuffled_y_true = y_true[permutation]

      for i in range(len(shuffled_interacted_pairs)):
        user_idx, item_idx = shuffled_interacted_pairs[i]
        y_true_sample = shuffled_y_true[i]
        y_pred = self.forward(user_idx, item_idx)
        loss = np.square(y_pred - y_true_sample)
        gradient_from_loss = 2 * (y_pred - y_true_sample) # Gradient of MSE

        self.backward(gradient_from_loss)

        epoch_loss += loss.item()

    final_avg_epoch_loss = epoch_loss / len(interacted_pairs)
    print(f'NCF training completed! Final Epoch Average Loss: {final_avg_epoch_loss:.6f}')

  def inference(self, user_ids, movie_ids, unobserveded_interactions):
    for i, (user_id, movie_id) in enumerate(unobserveded_interactions):
      predicted_rating = round(self.forward(user_id, movie_id).item(), 2)
      if predicted_rating > 0.80:
          self.recommendations.append((user_ids[user_id], movie_ids[movie_id], predicted_rating))

  def save_model(self):
    with open(self.filename, 'wb') as file:
      pickle.dump(self, file)


def normalize(ratings):
    return np.array(
        (ratings - ratings.min()) / 
        (ratings.max() - ratings.min()))


def get_interations(dataset):
    user_to_idx = {user_id: idx for idx, user_id in enumerate(dataset['userID'].unique())}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(dataset['movieID'].unique())}

    observeded_interactions = []
    for user, movie in dataset.iloc[:,:2].to_numpy():
        observeded_interactions.append([user_to_idx[user], movie_to_idx[movie]])

    unobserveded_interactions = []
    for user in dataset['userID'].unique():
        for movie in dataset['movieID'].unique():
            if (user, movie) not in zip(dataset['userID'], dataset['movieID']):
                unobserveded_interactions.append([user_to_idx[user], movie_to_idx[movie]])

    unobserveded_interactions = np.array(unobserveded_interactions)

    return observeded_interactions, unobserveded_interactions


def initialize_ncf_model(dataset):
    return NCF(
        total_users = dataset['userID'].unique().shape[0],
        total_items = dataset['movieID'].unique().shape[0],
        mlp_neurons_per_layer= [64, 32, 16, 8],
        learning_rate= 0.005,
        embedding_dim_gmf= 32,
        embedding_dim_mlp= 64
    )


def build_NCF_model(use_pretrained_model = True, row_limit = 5000):
    ncf_model = None
    model_name = 'NCF_model.pkl'
    
    if use_pretrained_model & os.path.isfile(model_name):
        with open(model_name, 'rb') as file:
            ncf_model = pickle.load(file)
    
    else:
        filename = 'data/user_ratedmovies.dat'
        dataset = pd.read_csv(
            filename, sep='\t', usecols=['userID', 'movieID', 'rating']).iloc[:row_limit]
        
        normalized_ratings = normalize(dataset['rating'])
        observeded_interactions, unobserveded_interactions = get_interations(dataset)
        
        ncf_model = initialize_ncf_model(dataset)
        ncf_model.train(observeded_interactions, normalized_ratings, epochs=50)
        
        ncf_model.inference(
            dataset['userID'].unique().tolist(), dataset['movieID'].unique().tolist(), unobserveded_interactions)

        ncf_model.save_model()
    return ncf_model