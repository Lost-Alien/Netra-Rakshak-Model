%% STEP 5: CLASS-WEIGHTED MODEL TRAINING (FIXES CLASS IMBALANCE)
clc; clear; close all;

% 1. Load Enhanced Dataset
datasetDir = 'D:\Lost_Projects\Netra_Rakshak_Model\enhanced_dataset\';

if ~exist(datasetDir, 'dir')
    error('❌ Cannot find enhanced_dataset folder! Verify the directory.');
end

fprintf('📦 Scanning enhanced dataset...\n');
imds = imageDatastore(datasetDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 2. Calculate Inverse Frequency Class Weights
labelCount = countEachLabel(imds);
disp('Current Image Distribution per Severity Level:');
disp(labelCount);

totalImages = sum(labelCount.Count);
numClasses = height(labelCount);

% Higher weights are assigned to underrepresented classes (Levels 3 and 4)
classWeights = totalImages ./ (numClasses * labelCount.Count);
classWeights = single(classWeights');

fprintf('⚖️ Computed Penalty Weights for Classes 0 to 4:\n');
for c = 1:numClasses
    fprintf('   Class %s: Weight = %.2f\n', char(labelCount.Label(c)), classWeights(c));
end

% 3. Partition Data (80% Train, 20% Validation)
fprintf('✂️ Splitting data into Training (80%%) and Validation (20%%)...\n');
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

imageSize = [224, 224, 3];
augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true);

augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'DispatchInBackground', true);

augImdsValidation = augmentedImageDatastore(imageSize, imdsValidation, ...
    'DispatchInBackground', true);

% 4. Build ResNet-50 Architecture with Weighted Cross-Entropy Loss
fprintf('🧠 Constructing network with class-weighted penalty loss...\n');
net = resnet50;
lgraph = layerGraph(net);

newFC = fullyConnectedLayer(numClasses, 'Name', 'DR_FC', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newClassification = classificationLayer('Name', 'DR_OutputClass', ...
    'Classes', labelCount.Label, ...
    'ClassWeights', classWeights);

lgraph = replaceLayer(lgraph, 'fc1000', newFC);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassification);

% 5. Hardware Training Options for RTX 5070 Ti & i9-14900K
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsValidation, ...
    'ValidationFrequency', 20, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

% 6. Train the Model
fprintf('🚀 Starting GPU-accelerated weighted training...\n');
[trainedNet, trainInfo] = trainNetwork(augImdsTrain, lgraph, options);

% 7. Overwrite Model File with the Calibrated Network
outputPath = 'D:\Lost_Projects\Netra_Rakshak_Model\trained_dr_classifier.mat';
save(outputPath, 'trainedNet');
fprintf('💾 SUCCESS: Balanced model saved to "%s"!\n', outputPath);