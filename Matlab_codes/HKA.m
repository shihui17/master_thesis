function sol=HKA(param,D,costfunc)
% param = [N,Nb_best,alpha, sd] parameters of the HKA
% N : number of points
% Nb_best : number of best candidates
% alpha : slowdown coefficient
% sd : if sd=1 test of the side constraints,
% no test if <>1
%
% D = [x1_min ... xn_min ; x1_max ... xn_max] search domain
% x1_min ... xn_min : lower bounds of the decision
% variables
% x1_max ... xn_max : upper bounds of the decision
% variables
%
% costfunc = string of matlab function to optimize
%
% sol = [costvalue, x1_opt ... xn_opt] best solution found
% costvalue : cost function value of the best solution
% x1_opt ... xn_opt : best solution found
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialization
rand('state',sum(100*clock)); % Init random generator
n=length(D(1,:)); % number of decision variables
mu=zeros(1,n); % mean vector of the Gaussian generator
sigma=zeros(1,n); % std vector of the Gaussian generator
sol=zeros(1,n+1); % vector solution
NbMaxIter=300; % max number of iteration for the stopping
% rule
N=param(1); % number of points
Nb_best=param(2); % number of best candidates
alpha=param(3); % slowdown coefficient
sd=param(4); % side constraints test
lb=repmat(D(1,:),N,1); % For the side constraints test lower
% bounds
ub=repmat(D(2,:),N,1); % For the side constraints test upper
% bounds
for i=1:n
    mu(i)=(D(1,i)+D(2,i))/2; % init of the mean vector
    sigma(i)=(D(2,i)-D(1,i))/6; % init of the std vector
end

NbIter=1;crit=inf;rho=inf;

while ((rho>0.005) && (NbIter<NbMaxIter))
    % generate sequence of vectors
    x=ones(N,1)*mu+randn(N,n)*diag(sigma);
    if sd==1 % side constraints test
    x=(x<lb).*lb+(x>lb).*x;
    x=(x>ub).*ub+(x<ub).*x;
    end
    % compute de cost function
    valcost=feval(costfunc,x);
    % sort lower to bigger
    sort_valcost=sortrows([x valcost],n+1);
    % compute xi
    xi=mean(sort_valcost(1:Nb_best,1:n));
    % compute the standard deviation vector
    sigma1=std(sort_valcost(1:Nb_best,1:n));
    % compute de variance vector
    V=sigma1.^2;
    % compute de Kalman gain
    L=sigma.^2./(sigma.^2.+V);
    % update the estimate of the optimum
    q=mu+L.*(xi-mu);
    % compute the variance vector of the posterior-estimation
    % error
    P=sigma.^2-L.*sigma.^2;
    % intermediate variable for slowdown factor
    s=min(1,mean(sigma1));
    % compute the slowdown factor
    a=alpha*(s^2/(s^2+max(sqrt(P))^2));
    % update the std of the Gaussian
    sigma=sigma+a*(sqrt(P)-sigma);
    % update the mean of the Gaussian
    mu=q;
    result=[feval(costfunc,mu) mu];
    % Save the best
    if result(1)<crit
        sol=result
        crit=result(1);
    end
    NbIter=NbIter+1;
    % stopping rule
    best=sort_valcost(1:Nb_best,1:n);
    for i=2:Nb_best
        dist(i)=sqrt((best(1,:)-best(i,:))*(best(1,:)-best(i,:))');
    end
    rho=max(dist);
end


