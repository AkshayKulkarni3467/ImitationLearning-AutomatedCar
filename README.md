DAGGER proceeds by collecting a dataset at each iteration under the current policy and trains the next policy under the aggregate of all collected datasets. This algorithm can be interpreted as a Follow-The-Leader 
algorithm in that at iteration n we pick the best policy πˆn+1 in hindsight, i.e. under all trajectories seen so far over the iterations.

![image](https://github.com/AkshayKulkarni3467/ImitationLearning-AutomatedCar/assets/129979542/e1c18d91-1f07-4b43-856f-df2f9e0fdbd2)

Here in this project, we solve the Car racing environment by manually providing action space for n iterations and training a CNN after every iteration. After every iteration, aggregate the supervised data with action spaces as labels to the i-1th dataset.

To train the algorithm:

```
python dagger.py

```

To test the models:

```
python test_agent.py --path dagger_models/{model_name}

```

Visualization of car after receiving human input for 1 iteration:

