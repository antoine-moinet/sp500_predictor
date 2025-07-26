This project tests different methods to predict future returns of the S&P500 database. We use a linear regression to predict returns of the next day based on current returns and Carhart factors.
Then we do the same with a feedforward NN. And then with a Recurrent NN, fed with past returns and factors over a lookback period.
The Models are then used to implement a simple trading strategy, going long on stocks we predict will perform well and short on stocks we predict will perform poorly.
