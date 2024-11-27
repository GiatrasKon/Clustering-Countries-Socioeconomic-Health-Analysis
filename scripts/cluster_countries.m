clear
format compact
close all
rand('seed',0)
randn('seed',0)

load data_country
X=Countrydata;
[N,l]=size(X);
clear Countrydata
DISPLAY_FIGURES = true;
PLOT_STYLES = ['or', '+g', 'sb', 'xy', 'vm'];
LABELS = {'child mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life expec', 'total fer', 'gdpp'};

% ========================= Feeling the data =========================

% Missing values per column
missing_values_cols = any(isnan(X), 1);
disp('Missing Values Along Columns:');
disp(missing_values_cols);

% Descriptive statistics for X
[X_min, X_max, X_mean, X_median, X_var, X_std, X_percentiles] = data_statistics(X);

% Finding corresponding country names
[country_min, country_median, country_max] = find_country_names(country, X, X_min, X_median, X_max);

% --------------------------------------------------------------------

% Display histograms
cols = size(X, 2);
for i = 1:cols
    current_col = X(:, i);
    if DISPLAY_FIGURES
        figure;
        histogram(current_col, 20);
        title(['Histogram of column: ', LABELS{i}]);
        xlabel(['Values in column: ', LABELS{i}]);
        ylabel('Frequency');
    end
end

% --------------------------------------------------------------------

% Display boxplots
for i = 1:cols
    current_col = X(:, i);
    if DISPLAY_FIGURES
        figure;
        boxplot(current_col);
        title(['Boxplot of column: ', LABELS{i}]);
    end
end

% --------------------------------------------------------------------

% Correlation coefficients heatmap
correlation_matrix = corrcoef(X);
if DISPLAY_FIGURES
    colormap('jet');
    figure;
    heatmap(correlation_matrix, 'XDisplayLabels', LABELS, 'YDisplayLabels', LABELS);
    title('Correlation Heatmap');
end

% --------------------------------------------------------------------

% Normalizing each row as a zero mean unit variance distributions
X_stand_norm = (X-ones(N,1)*mean(X)) ./ (ones(N,1)*std(X));
corr_stand_norm = corrcoef(X_stand_norm);
if DISPLAY_FIGURES
    colormap('jet');
    figure;
    heatmap(corr_stand_norm, 'XDisplayLabels', LABELS, 'YDisplayLabels', LABELS);
    title('Correlation Heatmap after normalization using unit variance distributions');
end

% Descriptive statistics for X_stand_norm
[X_stand_norm_min, X_stand_norm_max, X_stand_norm_mean, X_stand_norm_median, X_stand_norm_var, X_stand_norm_std, X_stand_norm_percentiles] = data_statistics(X_stand_norm);

% --------------------------------------------------------------------

% Normalizing each row in the min-max range
X_minmax_norm = (X-ones(N,1)*min(X)) ./ (ones(N,1)*(max(X)-min(X)));
corr_minmax_norm = corrcoef(X_minmax_norm);
if DISPLAY_FIGURES
    colormap('jet');
    figure;
    heatmap(corr_minmax_norm, 'XDisplayLabels', LABELS, 'YDisplayLabels', LABELS);
    title('Correlation Heatmap after normalization using min-max range');
end

% Descriptive statistics for X_minmax_norm
[X_minmax_norm_min, X_minmax_norm_max, X_minmax_norm_mean, X_minmax_norm_median, X_minmax_norm_var, X_minmax_norm_std, X_minmax_norm_percentiles] = data_statistics(X_minmax_norm);

% --------------------------------------------------------------------

% Difference of original data and normalized data
diff_stand_norm = abs(corr_stand_norm - correlation_matrix);
diff_stand_norm_min = min(min(diff_stand_norm));
diff_stand_norm_max = max(max(diff_stand_norm));

diff_minmax_norm = abs(corr_minmax_norm - correlation_matrix);
diff_minmax_norm_min = min(min(diff_minmax_norm));
diff_minmax_norm_max = max(max(diff_minmax_norm));

% ========================= Feature selection/transformation =========================

[coeff, score, ~, ~, explained] = pca(X_stand_norm);
cumulative = cumsum(explained);
components = find(cumulative' >= 95.0);

% EXPERIMENT A: reduce normalized data
% Retain the selected number of components
% GDPP, Inflation, Total Fertality, Life expectancy
% 9     6          8                7
X_exp_A = score(:, components);
labels_exp_A = LABELS([6, 7, 8, 9]);

% EXPERIMENT B: reduce normalized data  
% GDPP, Inflation, Total Fertality, Health, Life expectancy
% 9     6          8                3       7
X_exp_B = X_stand_norm(:, [3, 6, 7, 8, 9]);
labels_exp_B = LABELS([3, 6, 7, 8, 9]);

% ========================= Selection of the clustering algorithm =========================

if DISPLAY_FIGURES
    figure;
    grid on;
    hold on;
    for i = 1:4
        [f,xi] = ksdensity(X_exp_A(:,i));
        plot(xi, f, 'DisplayName', labels_exp_A{i});
    end
    hold off;
    legend('show');
    title('Kernel Density Estimates for Selected Features in Experiment A');
    drawnow;
end

if DISPLAY_FIGURES
    figure;
    grid on;
    hold on;
    for i = 1:5
        [f,xi] = ksdensity(X_exp_B(:,i));
        plot(xi, f, 'DisplayName', labels_exp_B{i});
    end
    hold off;
    legend('show');
    title('Kernel Density Estimates for Selected Features in Experiment B');
    drawnow;
end

% ========================= Execution of the clustering algorithm =========================

% Adapt the nunber of clusters
clusters_num = [3, 4, 5];
% clusters_num = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20];

% Seed number
seed_number = 200;

% Eval criterions (for evalclusters)
eval_criterions = {'CalinskiHarabasz', 'DaviesBouldin', 'silhouette'};

% ------------------------- k-means execution -------------------------

% EXPERIMENT A: execute clustering algorithms
[ theta_k_means_a, clusters_k_means_a, J_k_means_a, results_k_means_a] = execute_k_means(country, X_exp_A', clusters_num, seed_number);
exp_title = ['k-means, expA, seed ' num2str(seed_number)];
if DISPLAY_FIGURES
    display_silhouette(X_exp_A, clusters_k_means_a, ['Silhouette ' exp_title]);
    display_elbow_curve(clusters_num, J_k_means_a', ['Elbow Curve ' exp_title], '-om');
end
preview_eval_clusters(X_exp_A, clusters_k_means_a, eval_criterions)

% EXPERIMENT B: execute clustering algorithms
[ theta_k_means_b, clusters_k_means_b, J_k_means_b, results_k_means_b] = execute_k_means(country, X_exp_B', clusters_num, seed_number);
exp_title = ['k-means, expB, seed ' num2str(seed_number)];
if DISPLAY_FIGURES
    display_silhouette(X_exp_B, clusters_k_means_b, ['Silhouette ' exp_title]);
    display_elbow_curve(clusters_num, J_k_means_b', ['Elbow Curve ' exp_title], '-om');
end
preview_eval_clusters(X_exp_B, clusters_k_means_b, eval_criterions)

% ------------------------- k-medians execution -------------------------

% EXPERIMENT A: execute clustering algorithms
[ theta_k_medians_a, clusters_k_medians_a, J_k_medians_a, results_k_medians_a] = execute_k_medians(country, X_exp_A', clusters_num, seed_number);
exp_title = ['k-medians, expA, seed ' num2str(seed_number)];
if DISPLAY_FIGURES
    display_silhouette(X_exp_A, clusters_k_medians_a, ['Silhouette ' exp_title]);
    display_elbow_curve(clusters_num, J_k_medians_a', ['Elbow Curve ' exp_title], '-or');
end
preview_eval_clusters(X_exp_A, clusters_k_medians_a, eval_criterions)

% EXPERIMENT B: execute clustering algorithms
[ theta_k_medians_b, clusters_k_medians_b, J_k_medians_b, results_k_medians_b] = execute_k_medians(country, X_exp_B', clusters_num, seed_number);
exp_title = ['k-medians, expB, seed ' num2str(seed_number)];
if DISPLAY_FIGURES
    display_silhouette(X_exp_B, clusters_k_medians_b, ['Silhouette ' exp_title]);
    display_elbow_curve(clusters_num, J_k_medians_b', ['Elbow Curve ' exp_title], '-or');
end
preview_eval_clusters(X_exp_B, clusters_k_medians_b, eval_criterions)

% ------------------------- Re-running k-medians for optimal number of clusters (3) -------------------------

% Number of features in Experiment A
num_features_A = 4; 
feature_indices_A = 1:num_features_A; % Assuming features are the first 4
labels_A = labels_exp_A; % Assuming these are the labels for Experiment A

% Generate all combinations of 2 features for Experiment A
combinations_A = nchoosek(feature_indices_A, 2);

% Iterate over each combination for Experiment A
for i = 1:size(combinations_A, 1)
    % Select features for this combination
    features_2D_A = X_exp_A(:, combinations_A(i, :));

    % Running k-medians on these two features
    [theta_k_medians_a_2D, clusters_k_medians_a_2D, ~, ~] = execute_k_medians(country, features_2D_A', [3], seed_number);

    % Running k-medians for 4D data in Experiment A
    [theta_k_medians_a_4D, clusters_k_medians_a_4D, J_k_medians_a_4D, results_k_medians_a_4D] = execute_k_medians(country, X_exp_A', [3], seed_number);

    % Visualizing clustering results for Experiment A
    if DISPLAY_FIGURES
        figure; % Create a new figure for each pair of features
        set(gcf, 'Position', [100, 100, 1024, 512]); % Set a wider figure size for this figure

        % High-dimensional clustering (4D data)
        subplot('Position', [0.05, 0.1, 0.4, 0.8]); % Position as [left, bottom, width, height]
        scatter(X_exp_A(:, combinations_A(i, 1)), X_exp_A(:, combinations_A(i, 2)), 10, clusters_k_medians_a_4D, 'filled');
        title(['4D Clustering for Experiment A: ', labels_A{combinations_A(i, 1)}, ' & ', labels_A{combinations_A(i, 2)}]);
        xlabel(labels_A{combinations_A(i, 1)});
        ylabel(labels_A{combinations_A(i, 2)});
        axis square; % Make the current axes region square

        % 2D clustering for the same features
        subplot('Position', [0.55, 0.1, 0.4, 0.8]); % Position as [left, bottom, width, height]
        scatter(features_2D_A(:,1), features_2D_A(:,2), 10, clusters_k_medians_a_2D, 'filled');
        title(['2D Clustering for Experiment A: ', labels_A{combinations_A(i, 1)}, ' & ', labels_A{combinations_A(i, 2)}]);
        xlabel(labels_A{combinations_A(i, 1)});
        ylabel(labels_A{combinations_A(i, 2)});
        axis square; % Make the current axes region square

        drawnow;
    end
end

% Number of features in Experiment B
num_features_B = 5; 
feature_indices_B = 1:num_features_B; % Assuming features are the first 5
labels_B = labels_exp_B; % Assuming these are the labels for Experiment B

% Generate all combinations of 2 features for Experiment B
combinations_B = nchoosek(feature_indices_B, 2);

% Iterate over each combination for Experiment B
for i = 1:size(combinations_B, 1)
    % Select features for this combination
    features_2D_B = X_exp_B(:, combinations_B(i, :));

    % Running k-medians on these two features
    [theta_k_medians_b_2D, clusters_k_medians_b_2D, ~, ~] = execute_k_medians(country, features_2D_B', [3], seed_number);

    % Running k-medians for 5D data in Experiment B
    [theta_k_medians_b_5D, clusters_k_medians_b_5D, J_k_medians_b_5D, results_k_medians_b_5D] = execute_k_medians(country, X_exp_B', [3], seed_number);


    % Visualizing clustering results for Experiment B
    if DISPLAY_FIGURES
        figure; % Create a new figure for each pair of features
        set(gcf, 'Position', [100, 100, 1024, 512]); % Set a wider figure size for this figure
    
        % High-dimensional clustering (5D data)
        subplot('Position', [0.05, 0.1, 0.4, 0.8]); % Position as [left, bottom, width, height]
        scatter(X_exp_B(:, combinations_B(i, 1)), X_exp_B(:, combinations_B(i, 2)), 10, clusters_k_medians_b_5D, 'filled');
        title(['5D Clustering for Experiment B: ', labels_B{combinations_B(i, 1)}, ' & ', labels_B{combinations_B(i, 2)}]);
        xlabel(labels_B{combinations_B(i, 1)});
        ylabel(labels_B{combinations_B(i, 2)});
        axis square; % Make the current axes region square
    
        % 2D clustering for the same features
        subplot('Position', [0.55, 0.1, 0.4, 0.8]); % Position as [left, bottom, width, height]
        scatter(features_2D_B(:,1), features_2D_B(:,2), 10, clusters_k_medians_b_2D, 'filled');
        title(['2D Clustering for Experiment B: ', labels_B{combinations_B(i, 1)}, ' & ', labels_B{combinations_B(i, 2)}]);
        xlabel(labels_B{combinations_B(i, 1)});
        ylabel(labels_B{combinations_B(i, 2)});
        axis square; % Make the current axes region square
    
        drawnow;
    end
end

% ========================= Characterization of the Clusters =========================


% ------------------------- Experiment A -------------------------

% Identify the final cluster representatives
final_representatives_A = theta_k_medians_a_4D; % Final representatives

% Calculate detailed statistics for each feature within each cluster
% Preallocate statistics for each cluster
cluster_stats_A = cell(3, 1);

% Loop through each cluster to compute and store statistics
for k = 1:3
    % Extract the data points belonging to cluster k
    cluster_data = X_exp_A(clusters_k_medians_a_4D == k, :);
    
    % Get detailed statistics for each feature in cluster k
    [min_vals, max_vals, mean_vals, median_vals, var_vals, std_vals, percentiles] = data_statistics(cluster_data);
    
    % Combine the statistics into a matrix and store in the cell array
    cluster_stats_A{k} = [min_vals; max_vals; mean_vals; median_vals; var_vals; std_vals; percentiles];
end


% Plot PDFs for each feature in each cluster
for k = 1:3  
    if DISPLAY_FIGURES
        figure;
        grid on;
        hold on;
        % For each feature within the cluster
        for j = 1:num_features_A 
            % Extract the data for feature j within cluster k
            data_feature = X_exp_A(clusters_k_medians_a_4D == k, j);
            % Estimate and plot the PDF using ksdensity
            [f,xi] = ksdensity(data_feature);
            plot(xi, f, 'DisplayName', [labels_exp_A{j} ' - Cluster ' num2str(k)]);
        end
        hold off;
        legend('show');
        title(['Kernel Density Estimates for Cluster ' num2str(k) ' in Experiment A']);
        drawnow;
    end
end


% Compare the standard deviations and mean values for each feature of each cluster to the original standardized dataset
% Initialize the storage for the comparison metrics
cluster_comparison_A = cell(3, 1);

for k = 1:3
    % Extract the data for cluster k
    cluster_k_data = X_exp_A(clusters_k_medians_a_4D == k, :);
    
    % Calculate the statistics for cluster k
    [cluster_min, cluster_max, cluster_mean, cluster_median, cluster_var, cluster_std, cluster_percentiles] = data_statistics(cluster_k_data);
    
    % Calculate the differences from the original standardized data
    mean_difference = cluster_mean; % Since the original mean is 0 for all features
    std_difference = cluster_std - 1; % Since the original std is 1 for all features
    
    % Store the results
    cluster_comparison_A{k}.mean_difference = mean_difference;
    cluster_comparison_A{k}.std_difference = std_difference;
end


% List the countries belonging to each cluster
% Initialize a cell array to store the country names for each cluster
countries_in_clusters_A = cell(3, 1);

for k = 1:3
    % Find the indices of the countries belonging to the k-th cluster
    indices = find(clusters_k_medians_a_4D == k);
    
    % Store the country names in the cell array
    countries_in_clusters_A{k} = country(indices);
end


% ------------------------- Experiment B -------------------------

% Identify the final cluster representatives
final_representatives_B = theta_k_medians_b_5D; % Final representatives


% Calculate detailed statistics for each feature within each cluster
% Preallocate statistics for each cluster
cluster_stats_B = cell(3, 1);

% Loop through each cluster to compute and store statistics
for k = 1:3
    cluster_data = X_exp_B(clusters_k_medians_b_5D == k, :);
    [min_vals, max_vals, mean_vals, median_vals, var_vals, std_vals, percentiles] = data_statistics(cluster_data);
    cluster_stats_B{k} = [min_vals; max_vals; mean_vals; median_vals; var_vals; std_vals; percentiles];
end


% Plot the probability density functions (PDFs) of features for each cluster
for k = 1:3  
    if DISPLAY_FIGURES
        figure;
        hold on;
        for j = 1:num_features_B 
            data_feature = X_exp_B(clusters_k_medians_b_5D == k, j);
            [f,xi] = ksdensity(data_feature);
            plot(xi, f, 'DisplayName', [labels_exp_B{j} ' - Cluster ' num2str(k)]);
        end
        hold off;
        legend('show');
        title(['Kernel Density Estimates for Cluster ' num2str(k) ' in Experiment B']);
        drawnow;
    end
end


% Compare the standard deviations and mean values for each cluster to the original standardized dataset
cluster_comparison_B = cell(3, 1);

for k = 1:3
    cluster_k_data = X_exp_B(clusters_k_medians_b_5D == k, :);
    [cluster_min, cluster_max, cluster_mean, cluster_median, cluster_var, cluster_std, cluster_percentiles] = data_statistics(cluster_k_data);
    mean_difference = cluster_mean; % Since the original mean is 0 for all features
    std_difference = cluster_std - 1; % Since the original std is 1 for all features
    cluster_comparison_B{k}.mean_difference = mean_difference;
    cluster_comparison_B{k}.std_difference = std_difference;
end


% List the countries belonging to each cluster
countries_in_clusters_B = cell(3, 1);

for k = 1:3
    indices = find(clusters_k_medians_b_5D == k);
    countries_in_clusters_B{k} = country(indices);
end


% ========================= FUNCTIONS =========================

function [min_X, max_X, mean_X, median_X, var_X, std_X, percentiles_X] = data_statistics(X)
    min_X = min(X);
    max_X = max(X);
    mean_X = mean(X);
    median_X = median(X);
    var_X = var(X);
    std_X = std(X);
    percentiles = [25, 50, 75];
    percentiles_X = prctile(X, percentiles);
end

function [country_min, country_median, country_max] = find_country_names(country, X, min_X, median_X, max_X)
    for i = 1:size(X,2) % Loop through each feature
        [~, min_index] = min(X(:,i)); % Find index of min value for feature i
        [~, median_index] = min(abs(X(:,i)-median_X(i))); % Find index closest to median for feature i
        [~, max_index] = max(X(:,i)); % Find index of max value for feature i

        country_min(i) = country(min_index); % Find country for min value
        country_median(i) = country(median_index); % Find country for median value
        country_max(i) = country(max_index); % Find country for max value
    end
end

function [ theta, clusters_history, J_history, cluster_assignments] = execute_k_means(country, data, clusters_matrix, seed_number)
    cluster_assignments = cell(length(country), length(clusters_matrix) + 1);
    cluster_assignments(:, 1) = country; 
    clusters_history = zeros(length(data), length(clusters_matrix));
    J_history = zeros(length(clusters_matrix), 1);
    for index = 1:length(clusters_matrix)
        clusters_num = clusters_matrix(index);
        theta_init = rand_init(data, clusters_num, seed_number);
        [ theta, clusters, J ] = k_means(data, theta_init);
        J_history(index) = J;
        for i = 1:length(clusters)+1
            cluster_assignments(clusters == i, index+1) = num2cell(i);
            clusters_history(:,index) = clusters(:);
        end
    end
end

function [ theta, clusters_history, J_history, cluster_assignments] = execute_k_medians(country, data, clusters_matrix, seed_number)
    cluster_assignments = cell(length(country), length(clusters_matrix) + 1);
    cluster_assignments(:, 1) = country;
    clusters_history = zeros(length(data), length(clusters_matrix));
    J_history = zeros(length(clusters_matrix), 1);
    for index = 1:length(clusters_matrix)
        clusters_num = clusters_matrix(index);
        theta_init = rand_init(data, clusters_num, seed_number);
        [ theta, clusters, J ] = k_medians(data, theta_init);
        J_history(index) = J;
        for i = 1:clusters+1
            cluster_assignments(clusters == i, index+1) = num2cell(i);
            clusters_history(:,index) = clusters(:);
        end
    end
end

function [] = display_silhouette(data, clusters_history, title)
    num_columns = size(clusters_history, 2);
    plot_cols = int32(3);
    plot_rows = idivide(num_columns, plot_cols) + mod(num_columns, plot_cols);
    figure;
    for K=1:num_columns
         subplot(double(plot_rows), double(plot_cols), K)
         [s, h] = silhouette(data, clusters_history(:, K),'Euclidean');
         avg = mean(s);
         x = [avg avg];
         y = [0 500];
         line(x,y,'Color','red','LineStyle','--')
         text(avg - 0.08 , -2, num2str(avg,2))
         text(-0.2, -7, title)
    end
end

function [] = display_elbow_curve(clusters_matrix, J_history, plot_title, plot_style)
    distortion_score = mean(J_history);
    [~, closest_index] = min(abs(J_history - distortion_score));
    closest_value = J_history(closest_index);
    elbow_point = find_elbow_point(clusters_matrix, J_history);
    optimal_K = elbow_point;
    clusters_local_min = clusters_matrix(find(islocalmin(J_history),1));
    if clusters_local_min < optimal_K
        optimal_K = clusters_local_min;
    end
    figure;
    plot(clusters_matrix, J_history, plot_style);
    xline(optimal_K, '-b')
    xlabel('Number of clusters');
    ylabel('Sum of squared distances');
    title(plot_title);
    legend('Sum of sq dist', ['elbowatk = ' num2str(optimal_K) ',score = ' num2str(closest_value)])
    drawnow;
end

function optimal_k = find_elbow_point(k_values, sum_of_distances)
    diff1 = diff(sum_of_distances);
    diff2 = diff(diff1);
    [~, idx] = max(diff2);
    optimal_k = k_values(idx + 1);
end

function [] = preview_eval_clusters(data, clusters_history, eval_criterions)
    for criterion=1:size(eval_criterions, 2)
        try
            eva = evalclusters(data, clusters_history, eval_criterions(criterion));
            disp(eva)
        catch
            disp('Error: at least one cluster contains 0 elements.')
        end
    end
end
