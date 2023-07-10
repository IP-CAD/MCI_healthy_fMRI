function Out = logm1(X)

[V D] = eig(X);
Out = V*diag(log(diag(D)))*V';