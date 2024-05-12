from sklearn.datasets import make_circles
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import torch
from torch import nn

# Split data into train and test sets
from sklearn.model_selection import train_test_split

from CircleModelV0 import CircleModelV0
from CircleModelV1 import CircleModelV1

from helper_functions import plot_decision_boundary,plot_predictions

# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values
# print(f"print first 10 Xs:{X[:5]}")
# print(f"print first 10 ys:{y[:5]}")

Circles = pd.DataFrame({"X1": X[:,0],"X2": X[:,1],"y":y})
Circles.head(10)
# display(Circles)
Circles.to_csv('mycsv.csv')

# Visualize with a plot

# plt.scatter(x=X[:, 0], 
#             y=X[:, 1], 
#             c=y, 
#             cmap=plt.cm.RdYlBu)

# plt.show()
# print(y.shape)
# print(X.shape)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)


model_0 = CircleModelV0().to(device)

with torch.inference_mode():
    untrained_preds = model_0(X_test)
# print(untrained_preds)
# plt.scatter(x=X_train[:, 0], 
#             y=X_train[:, 1], 
#             c=y_train, 
#             cmap=plt.cm.RdYlBu)

# plt.show()

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.01)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

y_logits = model_0(X_test.to(device))
# print(y_logits[:10])
# print(len(y_logits))

y_pred_probs = torch.sigmoid(y_logits)

y_preds_probs_label = torch.round(y_pred_probs)
# print(y_preds_probs_label.squeeze() )

torch.cuda.manual_seed(42)
torch.manual_seed(42)

epochs = 10000
for epoch in range(epochs):
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
