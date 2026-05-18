# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.


## Problem Statement and Dataset

### Problem Statement
The objective of this experiment is to develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical stock market data. The model learns patterns from previous stock closing prices and predicts future stock prices based on sequential dependencies in the data.

### Dataset
<img width="702" height="571" alt="image" src="https://github.com/user-attachments/assets/90dd61e4-8db6-4abf-bc4b-77e1209d49e0" />



## DESIGN STEPS

### STEP 1:
Import the required libraries such as **NumPy, Pandas, Matplotlib, Scikit-learn, and PyTorch** for data preprocessing, visualization, and deep learning model implementation.

### STEP 2:
Load the training and testing datasets from Google Drive and extract the **closing stock prices** from the dataset.

### STEP 3:
Normalize the stock price data using **MinMaxScaler** and create sequential data samples with a sequence length of **60 days** to capture temporal dependencies.

### STEP 4:
Convert the prepared sequences into **PyTorch tensors** and create a **DataLoader** for batch processing during training.

### STEP 5:
Define the **Recurrent Neural Network (RNN)** model with the following configuration:

- **Input Size** = 1  
- **Hidden Size** = 64  
- **Number of Layers** = 2  
- **Output Size** = 1  

The final hidden state is passed through a fully connected layer to predict stock prices.

### STEP 6:
Train the model using **Mean Squared Error (MSE)** loss function and **Adam optimizer**, evaluate the model on test data, and compare the **actual stock prices** with **predicted stock prices** using graphical visualization.

---

## PROGRAM

### Name: Vikamuhan Reddy

### Register Number: 212223240181

``` python
# Define RNN Model
class RNNModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):
        out, _ = self.rnn(x)

        out = self.fc(
            out[:, -1, :]
        )

        return out


# Train the Model

model = RNNModel()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=30
):
    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f'Epoch [{epoch+1}/{epochs}], '
            f'Loss: {total_loss/len(train_loader):.4f}'
        )

train_model(
    model,
    train_loader,
    criterion,
    optimizer
)
```
---

## OUTPUT

### Training Loss Over Epochs Plot
<img width="630" height="498" alt="Screenshot 2026-05-18 143350" src="https://github.com/user-attachments/assets/16e1b020-e488-48aa-a8b8-8e953de18a24" />



### True Stock Price, Predicted Stock Price vs Time
<img width="880" height="563" alt="Screenshot 2026-05-18 143435" src="https://github.com/user-attachments/assets/ef5d49a3-74d3-4f10-ab1a-4bc6c9977744" />


### Predictions

<img width="300" height="46" alt="Screenshot 2026-05-18 143542" src="https://github.com/user-attachments/assets/dbf74159-5b21-4c91-b547-48986415001a" />



## RESULT

Thus, a Recurrent Neural Network (RNN) model for stock price prediction was successfully developed using historical stock closing price data. The model was trained and tested successfully, and the predicted stock prices were compared with actual stock prices to evaluate the model performance.
