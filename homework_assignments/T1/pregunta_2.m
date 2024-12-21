clear all;

simulations_ar2 = readtable('simulations_ar2.csv');

% disp('First rows of AR(2) simulations:');
% disp(head(simulations_ar2));

filtered_ar2 = simulations_ar2(year(simulations_ar2.Time) <= 2018, :);

simulations_ar1 = readtable('simulations_ar1.csv');

% disp('First rows of AR(1) simulations:');
% disp(head(simulations_ar1));

filtered_ar1 = simulations_ar1(year(simulations_ar1.Time) <= 2018, :);


%% AR(2)

tic;

length_aic_ar2 = [];
length_bic_ar2 = [];

% Loop through each simulation
for sim_idx = 1:size(filtered_ar2, 2) - 1
    time_series = filtered_ar2{:, sim_idx + 1}; % Exclude the 'Time' column

    pmax = 24;

    aic_order = LagOrderSelectionARp(time_series,pmax,"AIC");
    bic_order = LagOrderSelectionARp(time_series,pmax,"BIC");

    length_aic_ar2 = [length_aic_ar2; aic_order];
    length_bic_ar2 = [length_bic_ar2; bic_order];

end

end_time_ar2 = toc;
fprintf('Duration execution Loops AIC/BIC for AR(2): %.2f seconds\n', end_time_ar2);

% Save AR(2) results
save('ar2_results.mat', 'length_aic_ar2', 'length_bic_ar2', 'end_time_ar2');

writetable(array2table(length_aic_ar2), 'length_aic_ar2.csv');
writetable(array2table(length_bic_ar2), 'length_bic_ar2.csv');



%% AR(1)

tic;

length_aic_ar1 = [];
length_bic_ar1 = [];

% Loop through each simulation
for sim_idx = 1:size(filtered_ar1, 2) - 1
    time_series = filtered_ar1{:, sim_idx + 1}; % Exclude the 'Time' column

    pmax = 24;

    aic_order = LagOrderSelectionARp(time_series,pmax,"AIC");
    bic_order = LagOrderSelectionARp(time_series,pmax,"BIC");

    length_aic_ar1 = [length_aic_ar1; aic_order];
    length_bic_ar1 = [length_bic_ar1; bic_order];

end


end_time_ar1 = toc;
fprintf('Duration execution Loops AIC/BIC for AR(1): %.2f seconds\n', end_time_ar1);

% Save AR(1) results
save('ar1_results.mat', 'length_aic_ar1', 'length_bic_ar1', 'end_time_ar1');

writetable(array2table(length_aic_ar1), 'length_aic_ar1.csv');
writetable(array2table(length_bic_ar1), 'length_bic_ar1.csv');



