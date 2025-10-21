clc;
clear all;


% Question 1

%% ------------------------------- 1.1 ------------------------------------
T = 50000;
u = rand(T, 1);

%% ------------------------------- 1.2 ------------------------------------
mean_u = mean(u);

%% ------------------------------- 1.3 ------------------------------------
s = zeros(T, 1); 
s(1) = 0;      

% Probabilities
prob_stay_normal = 0.95; 
prob_stay_recession = 0.4;

% Stochastic process :
for t = 2:T
    if s(t-1) == 0
        if u(t) <= prob_stay_normal 
            s(t) = 0;
        else
            s(t) = 1;
        end
    else
        if s(t-1) == 1
            if u(t) <= prob_stay_recession 
                s(t) = 1;
            else 
                s(t) = 0;
            end                                         
        end
    end
end

%% ------------------------------- 1.4 ------------------------------------
proportion_recession = mean(s); 

%% ------------------------------- 1.5 ------------------------------------
lambda = 0.25; 
y = zeros(T, 1); 
y(1) = s(1); 

for t = 2:T
    if t <= 10
        y(t) = y(t-1) + lambda * (s(t-1) - y(t-1));
    else
        y(t) = y(t-1) + lambda * (sum(s(t-10:t-1) - y(t-10:t-1)) / 10);
    end
end

%% ------------------------------- 1.6 ------------------------------------
t_plot = 400:500; 

% Creating the plot
figure;
plot(t_plot, s(t_plot), 'b-', 'DisplayName', 'Real State'); 
hold on;
plot(t_plot, y(t_plot), 'r--', 'DisplayName', 'Expected State'); 
hold off;


title('Real vs Expected State of the Economy');
xlabel('Time (t)');
ylabel('State (0 = normal | 1 = recession)');
legend('show');


saveas(gcf, 'Q1_1.pdf');

%% ------------------------------- 1.7 ------------------------------------
rmse = sqrt(mean((s - y).^2)); 

%% ------------------------------- 1.8 ------------------------------------
lambda_values = [0.25, 0.3, 0.35, 0.4];
rmse_values = zeros(length(lambda_values), 1); 

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    y = zeros(T, 1); 
    y(1) = s(1);

    for t = 2:T
        if t <= 10
            y(t) = y(t-1) + lambda * (s(t-1) - y(t-1));
        else
            y(t) = y(t-1) + lambda * (sum(s(t-10:t-1) - y(t-10:t-1)) / 10);
        end
    end

 
    rmse_values(i) = sqrt(mean((s - y).^2)); 
end


[~, minIndex] = min(rmse_values);
plot(lambda_values, rmse_values, 'r-o'); 
hold on;

plot(lambda_values(minIndex), rmse_values(minIndex), 'g.', ...
    'MarkerSize', 20);


title('RMSE as a Function of Lambda Values');
xlabel('Lambda');
ylabel('RMSE');
hold off;

saveas(gcf, 'Q1_2.pdf');



% Answer : Most accurate expectations for lambda = 0.25