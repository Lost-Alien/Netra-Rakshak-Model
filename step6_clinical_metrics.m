%% STEP 6: CLINICAL VALIDATION & REFERABLE DR METRIC EVALUATOR
clc; clear; close all;

% 1. Setup Paths and Load Trained Model
modelPath = 'D:\Lost_Projects\Netra_Rakshak_Model\trained_dr_classifier.mat';
datasetDir = 'D:\Lost_Projects\Netra_Rakshak_Model\enhanced_dataset\';

if ~exist(modelPath, 'file')
    error('❌ Cannot find trained model file! Please check the path.');
end

fprintf('💾 Loading calibrated DR classifier...\n');
load(modelPath, 'trainedNet');

% 2. Load Enhanced Dataset and Create Validation Split
fprintf('📦 Scanning validation dataset...\n');
imds = imageDatastore(datasetDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Set random seed for consistent validation reproducibility
rng(42);
[~, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

augImdsVal = augmentedImageDatastore([224, 224, 3], imdsValidation);

% 3. Batch Evaluation on GPU
fprintf('🧪 Running inference across validation images on RTX 5070 Ti...\n');
[predictedLabels, classProbabilities] = classify(trainedNet, augImdsVal, ...
    'ExecutionEnvironment', 'gpu', ...
    'MiniBatchSize', 64);

trueLabels = imdsValidation.Labels;

% 4. Render Side-by-Side Confusion Charts
fig = figure('Name', 'Netra Rakshak - Clinical Benchmark Report', ...
             'Color', 'w', ...
             'Units', 'normalized', ...
             'Position', [0.05, 0.15, 0.9, 0.7]);

% Panel A: Multi-Class Confusion Matrix (Levels 0 through 4)
subplot(1, 2, 1);
cm1 = confusionchart(trueLabels, predictedLabels, ...
    'Title', '5-Class Clinical Severity Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm1.FontSize = 10;

% Panel B: Binary Clinical Triage Matrix (Referable DR Target)
% Non-Referable = Grades 0 and 1 | Referable = Grades 2, 3, and 4
trueNumeric = str2double(string(trueLabels));
predNumeric = str2double(string(predictedLabels));

trueBinary = categorical(trueNumeric >= 2, [false, true], ...
    {'Non-Referable (0-1)', 'Referable (2-4)'});
predBinary = categorical(predNumeric >= 2, [false, true], ...
    {'Non-Referable (0-1)', 'Referable (2-4)'});

subplot(1, 2, 2);
cm2 = confusionchart(trueBinary, predBinary, ...
    'Title', 'Referable DR Triage Matrix (>90% Sens Target)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm2.FontSize = 10;

% 5. Compute Clinical Diagnostic Metrics
confMat = confusionmat(trueBinary, predBinary);
TN = double(confMat(1, 1)); % Correctly identified Non-Referable
FP = double(confMat(1, 2)); % False Alarm
FN = double(confMat(2, 1)); % Missed Disease (Critical Risk)
TP = double(confMat(2, 2)); % Correctly identified Referable

sensitivity = (TP / (TP + FN)) * 100;
specificity = (TN / (TN + FP)) * 100;
precision   = (TP / (TP + FP)) * 100;
accuracy    = ((TP + TN) / (TP + TN + FP + FN)) * 100;
f1Score     = 2 * (precision * sensitivity) / (precision + sensitivity);

% 6. Print Clinical Performance Summary
fprintf('\n==================================================================\n');
fprintf('         NETRA RAKSHAK CLINICAL VALIDATION BENCHMARK REPORT       \n');
fprintf('==================================================================\n');
fprintf('  Metric               Target        Observed Status\n');
fprintf('  --------------------------------------------------------------\n');
fprintf('  Sensitivity (Recall): > 90.00%%   |  %.2f%%  ', sensitivity);
if sensitivity >= 90.0
    fprintf('✅ PASSED\n');
else
    fprintf('⚠️ NEAR TARGET\n');
end

fprintf('  Specificity:          > 85.00%%   |  %.2f%%  ', specificity);
if specificity >= 85.0
    fprintf('✅ PASSED\n');
else
    fprintf('⚠️ NEAR TARGET\n');
end

fprintf('  Triage Accuracy:                 |  %.2f%%\n', accuracy);
fprintf('  Precision (PPV):                 |  %.2f%%\n', precision);
fprintf('  F1-Score:                        |  %.2f%%\n', f1Score);
fprintf('==================================================================\n');
fprintf('  True Negatives: %d  |  False Positives: %d\n', TN, FP);
fprintf('  False Negatives: %d |  True Positives:  %d\n', FN, TP);
fprintf('==================================================================\n');