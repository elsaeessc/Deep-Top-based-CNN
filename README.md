# Deep-Top-based-CNN
This is the final setup to train a Top Tagger CNN. My dataloader is shit so feel free to write a better data loader. 
The Top Tagger CNN should distinguish between hadronic tt-bar events and QCD di-jet events. These events are simulated.
A explanation of the simulation is given in "Deep-learning Top Taggers or The End of QCD?" (https://arxiv.org/abs/1701.08784).
The event samples can be found in https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit .
These samples are not preprocessed. For this Top Tagger CNN I used the preprocessing script which is also here stored. 
After preprocessing you can use script and the preprocessed data to train a Top Tagger CNN.
After training evaluate the models and choose the model with the highest validation AUC.
The performance of the chosen model has to be measured with the preprocessed test data sample.
The model I trained and used for my analysis is also here stored. You can open it with Pytorch and Cuda.
Use therefore just following commands of Pytorch!:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

            # Convolutional Architecture
            class CNN(nn.Module):
                def __init__(self):
                    super(CNN, self).__init__()

                    self.padd = nn.ZeroPad2d((1, 2, 1, 2))

                    self.conv1 = nn.Conv2d(1, 128, 4, padding = False)
                    self.conv2 = nn.Conv2d(128, 64, 4, padding = False)
                    self.conv3 = nn.Conv2d(64, 64, 4, padding = False)

                    self.pool = nn.MaxPool2d(2)

                    self.fc1 = nn.Linear(10 * 10 * 64, 64) 
                    self.fc2 = nn.Linear(64, 256)
                    self.fc3 = nn.Linear(256, 256)
                    self.fc4 = nn.Linear(256, 2)

                def forward(self, x):

                    x = F.relu(self.conv1(self.padd(x)))
                    x = F.relu(self.conv2(self.padd(x)))
                    x = self.pool(x)

                    x = F.relu(self.conv3(self.padd(x)))
                    x = F.relu(self.conv3(self.padd(x)))
                    x = self.pool(x)

                    x = x.view(-1, 10 * 10 * 64)
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = F.relu(self.fc3(x))
                    x = nn.Softmax(dim=1)(self.fc4(x))

                    return x

            model = CNN().to(device)
            checkpoint = torch.load(f'model.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
