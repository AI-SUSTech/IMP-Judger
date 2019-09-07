%**************************************************************************************************
%Author: Peng Yang
%Last Edited: June 25, 2019 by Peng Yang
%Email: yangp@sustech.edu.cn
%Reference: Ke Tang, Peng Yang and Xin Yao. Negatively Correlated Search. IEEE Journal on Selected Areas in Communications, in press.

%**************************************************************************************************


clc;
clear all;
format long;
format compact;
disp('The newPNES runs!');

% Choose the problems to be tested. Please note that for test functions F7
% and F25, the global optima are out of the initialization range. For these
% two test functions, we do not need to check whether the variable violates
% the boundaries during the evolution after the initialization.
problemSet = [6:25];
for problemIndex = 1 : 15
    % Configuration of the problem
    problem = problemSet(problemIndex); % the index of the problem
    D = 30; % the dimension of the problem
    [o, A, M, a, alpha, b, lu] = DataLoading(problem, D); % Pre-load the data for the problem 
    
    % Configuration of NCS
    lambda = ceil(log(D)); % the number of search processes
    mu = 4+floor(3*log(D)); % the number of solutions in each search process
    phi_init = 0.00005; % the trade-off parameter betwen f and d
    eta_m_init = 1; % the step size of gradient descent for mean vector
    eta_c_init = (3+log(D))/(5*sqrt(D));% the step size of gradient descent for mean vector
    vl = repmat(lu(:,1), 1, mu);
    vu = repmat(lu(:,2), 1, mu);
    
    % Configuration of the test protocol
    MAXFES = 10000*D; % the total FE of each run
    totalTime = 25; % the total number of runs

    % Record the best results for each problem
    outcome = ones(totalTime,1).*1e300;

    % Definition of the structure of search processes
    sp = repmat(struct('x',zeros(D,mu),'fit',zeros(1,mu),'mean',zeros(D,1),'cov',zeros(D,1)), lambda, 1);

    time = 1;
    disp(['The ' num2str(problem) ' th problem started!******************']);
    
    while time <= totalTime
        
        % Set the randomness seed
        rand('seed', sum(100*clock));

        % Re-initialize the best solution recorder in this run
        min_f = 1e300;
        FES = 0;

        % Initialize the lambda search processes
        for i = 1 : lambda
            % Model the search process as Gaussian probabilistic distribution 
            sp(i).mean = lu(:,1) + rand(D,1) .* (lu(:,2) - lu(:,1));
            sp(i).cov = (lu(:,2)-lu(:,1))./lambda;
%             sp(i).cov = ones(D,1);
        end 

        % The main loop body
        while FES <MAXFES
            
            eta_m = eta_m_init .* ((exp(1)-exp(FES/MAXFES))/(exp(1)-1));
            eta_c = eta_c_init .* ((exp(1)-exp(FES/MAXFES))/(exp(1)-1));
            
            for i = 1 : lambda
                % Generate mu solutions for each search process
                sp(i).x = repmat(sp(i).mean,1,mu) + randn(D,mu) .* (repmat(sp(i).cov,1,mu));

                % Boundary checking and repairing
                if problem ~= 7 && problem ~= 25    
                    pos = sp(i).x < vl;
                    sp(i).x(pos) = 2 .* vl(pos) - sp(i).x(pos);     
                    pos = sp(i).x > vu;
                    sp(i).x(pos) = 2 .* vu(pos) - sp(i).x(pos);
                    pos= sp(i).x < vl;
                    sp(i).x(pos) = vl(pos);        
                end

                % Fitness evalution for mu solutions
                sp(i).fit = benchmark_func(sp(i).x, problem, o, A, M, a, alpha, b);
                FES = FES + mu;

                % Update the best solution ever found
                min_f = min(min(sp(i).fit), min_f);

                % Rank mu solutions ascendingly in terms of fitness
                [~,order] = sort(sp(i).fit); 
                [~,rank] = sort(order);

                % Set utility value for mu solutions in terms of rank
                tempU = max(0,log(mu/2+1)-log(transpose(rank)));
                utility = tempU./sum(tempU)-1/mu;

                % Prepare for calculating gradients (for saving computation time)
                invCov_i = 1./sp(i).cov;
                difXtoMean = sp(i).x-repmat(sp(i).mean,1,mu);

                % Calculate the gradients of expectation of fitness values    
                deltaMean_f = invCov_i.*mean(difXtoMean.*repmat(utility,D,1),2); % w.r.t. mean vector
                deltaCov_f = invCov_i.^2.*mean(difXtoMean.^2.*repmat(utility,D,1),2)./2; % w.r.t. covariance matrix

                % Calculate the gradients of distribution distances                
                deltaMean_d = zeros(D,1); % w.r.t. mean vector
                deltaCov_d = zeros(D,1); % w.r.t. covariance matrix
                for j = 1 : lambda
                    temp1 = 1./(sp(i).cov+sp(j).cov)./2;
                    temp2 = temp1.*(sp(i).mean-sp(j).mean);
                    deltaMean_d = deltaMean_d+temp2./4;
                    deltaCov_d = deltaCov_d+(temp1-temp2.^2./4-invCov_i)./4;
                end

                % Calculate the Fisher information 
                meanFisher = invCov_i.^2.*mean(difXtoMean.^2,2); % w.r.t. mean vector
                covFisher = mean((repmat(invCov_i.^2,1,mu).*difXtoMean.^2-repmat(invCov_i,1,mu)).^2,2)./4; % w.r.t. covariance matrix
               
                % Update the probilistic model of the search process
                sp(i).mean = sp(i).mean+1./meanFisher.*(deltaMean_f+deltaMean_d.*phi_init).*eta_m; % w.r.t. mean vector
                sp(i).cov = sp(i).cov+1./covFisher.*(deltaCov_f+deltaCov_d.*phi_init).*eta_c; % w.r.t. covariance matrix
                
                % Boundary checking and repairing for mean vectors 
                if problem ~= 7 && problem ~= 25
                    pos = sp(i).mean < lu(:,1);
                    sp(i).mean(pos) = 2 .* lu(pos,1) - sp(i).mean(pos);
                    pos = sp(i).mean > lu(:,2);
                    sp(i).mean(pos) = 2 .* lu(pos,2) - sp(i).mean(pos);
                    pos = sp(i).mean < lu(:,1);
                    sp(i).mean(pos) = lu(pos,1); 
                end
            end

            % Print the best solution ever found to the screen
            disp(['The best result at the ' num2str(FES) ' th FE is ' num2str(min_f)]);
        end
        outcome(time) = min_f;
        time = time + 1;
    end
    disp(['the ' num2str(problem) 'th problem result is:']);
    disp(['the mean result is: ' num2str(mean(outcome)) ' and the std is ' num2str(std(outcome))]);
end


