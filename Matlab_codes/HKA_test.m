D = [1 4 5 9 2; 3 6 12 20 22];
N = 10;
lb = repmat(D(1,:), N, 1);
ub = repmat(D(2,:), N, 1);

for i=1:n
    mu(i)=(D(1,i)+D(2,i))/2; % init of the mean vector
    sigma(i)=(D(2,i)-D(1,i))/6; % init of the std vector
end

x = ones(N,1)*mu
y = randn(N, n) * diag(sigma)
x+y