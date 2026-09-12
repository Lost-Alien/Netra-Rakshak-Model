%% STEP 3: DEEP LEARNING SEVERITY GRADING CLASSIFIER (RTX 5070 Ti OPTIMIZED)
clc; clear; close all;

% 1. Point to your perfectly enhanced dataset
datasetDir = 'D:\Lost_Projects\Netra_Rakshak_Model\enhanced_dataset\';

fprintf('📦 Loading enhanced retinal dataset into MATLAB memory...\n');
% imageDatastore automatically scans folders 0 to 4 and labels the data
imds = imageDatastore(datasetDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Count how many images are in each class to verify balance
labelCount = countEachLabel(imds);
disp(labelCount);

% 2. Split dataset: 80% for training your AI, 20% for testing its accuracy
fprintf('✂️ Splitting data into Training (80%%) and Validation (20%%) sets...\n');
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% 3. Real-Time Data Augmentation (Prevents AI from memorizing images)
% ResNet-50 requires input images to be exactly 224x224 pixels.
% We also randomly flip images to simulate left vs right eyes.
imageSize = [224 224 3];
augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true);

augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', augmenter);
augImdsValidation = augmentedImageDatastore(imageSize, imdsValidation);

% 4. Load Pre-trained ResNet-50 Architecture (Transfer Learning)
fprintf('🧠 Initializing ResNet-50 Neural Network layers...\n');
net = resnet50;
lgraph = layerGraph(net);

% Replace the final layers of ResNet (which were made for generic objects) 
% with layers specifically tracking our 5 Diabetic Retinopathy classes (0-4).
newLearnableLayer = fullyConnectedLayer(5, 'Name', 'DR_FC', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newClassificationLayer = classificationLayer('Name', 'DR_OutputClass');

lgraph = replaceLayer(lgraph, 'fc1000', newLearnableLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassificationLayer);

% 5. Configure Heavy-Duty Hardware Training Parameters (RTX 5070 Ti Tuning)
fprintf('🚀 Tuning execution environment for your NVIDIA RTX 5070 Ti...\n');
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...         % Uses healthy chunks of VRAM for training stability
    'MaxEpochs', 8, ...             % Number of full passes through the dataset
    'InitialLearnRate', 1e-4, ...   % Slow, careful learning rate for medical accuracy
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsValidation, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ... % Launches live, interactive graph window
    'ExecutionEnvironment', 'gpu');  % FORCES training onto your graphics card

% 6. UNLEASH THE AI TRAINING
fprintf('🔥 Training Started! Check the interactive window for real-time progress.\n');
[trainedNet, trainInfo] = trainNetwork(augImdsTrain, lgraph, options);

% 7. Save your trained model to your disk so you don't lose progress
fprintf('💾 Saving finalized AI model to disk...\n');
save('D:\Lost_Projects\Netra_Rakshak_Model\trained_dr_classifier.mat', 'trainedNet');
fprintf('🎉 SUCCESS: Model saved as "trained_dr_classifier.mat"!\n');
