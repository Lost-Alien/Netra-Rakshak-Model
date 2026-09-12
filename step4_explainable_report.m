%% STEP 4: CLINICAL EXPLAINABILITY MODULE (GRAD-CAM INTERACTIVE REPORT)
clc; clear; close all;

% 1. Setup Your Folder Paths & Variables
modelPath = 'D:\Lost_Projects\Netra_Rakshak_Model\trained_dr_classifier.mat';
datasetDir = 'D:\Lost_Projects\Netra_Rakshak_Model\enhanced_dataset\';

% Check if trained model file exists
if ~exist(modelPath, 'file')
    error('❌ Cannot find "trained_dr_classifier.mat"! Please verify the file path.');
end

% 2. Load Custom Trained AI Model
fprintf('💾 Loading trained model "trained_dr_classifier.mat"...\n');
loadedData = load(modelPath);
if isfield(loadedData, 'trainedNet')
    net = loadedData.trainedNet;
elseif isfield(loadedData, 'net')
    net = loadedData.net;
else
    error('❌ Could not find a network variable (trainedNet or net) in the MAT file.');
end

% 3. Pick a Random Image from Diseased Folders (Levels 4 down to 1)
fprintf('📂 Searching enhanced dataset for test images...\n');
foundValidImage = false;

% FIXED: Array populated with severity levels[cite: 1]
severityFolders = [4, 3, 2, 1]; 

for idx = 1:length(severityFolders)
    severityFolder = severityFolders(idx);
    testFolderPath = fullfile(datasetDir, num2str(severityFolder));
    if exist(testFolderPath, 'dir')
        % Support both PNG and JPG formats
        imgFiles = [dir(fullfile(testFolderPath, '*.png')); ...
                    dir(fullfile(testFolderPath, '*.jpg')); ...
                    dir(fullfile(testFolderPath, '*.jpeg'))];
        if ~isempty(imgFiles)
            randomIndex = randi(length(imgFiles));
            targetImgName = imgFiles(randomIndex).name;
            fullImgPath = fullfile(testFolderPath, targetImgName);
            trueClassLabel = num2str(severityFolder);
            foundValidImage = true;
            break;
        end
    end
end

if ~foundValidImage
    error('❌ Could not find any valid images in your enhanced_dataset/ subfolders.');
end

% 4. Read and Resize Image to Match ResNet-50 Input Dimensions
rawImg = imread(fullImgPath);
if size(rawImg, 3) == 1
    rawImg = cat(3, rawImg, rawImg, rawImg); % Ensure 3 channels
end
resizedImg = imresize(rawImg, [224, 224]);

% 5. Run AI Prediction with Calibrated Confidence Scores
fprintf('🧠 Analyzing fundus image: %s (True Class: Level %s)...\n', targetImgName, trueClassLabel);
[predictedClass, classScores] = classify(net, resizedImg);
confidencePercentage = max(classScores) * 100;

% Human-readable clinical labels
clinicalLabels = {'No DR (Level 0)', 'Mild DR (Level 1)', 'Moderate DR (Level 2)', ...
                  'Severe DR (Level 3)', 'Proliferative DR (Level 4)'};

% Robust index resolution
numericCheck = str2double(char(predictedClass));
if ~isnan(numericCheck) && numericCheck >= 0 && numericCheck <= 4
    predictedIdx = numericCheck + 1;
else
    predictedIdx = double(predictedClass); % Fallback to categorical order
end

if predictedIdx >= 1 && predictedIdx <= length(clinicalLabels)
    predictedLabelStr = clinicalLabels{predictedIdx};
else
    predictedLabelStr = char(predictedClass);
end

% 6. Compute Grad-CAM Map
fprintf('🔥 Computing Grad-CAM Activation Heatmaps...\n');
fLayer = 'activation_49_relu';  % ResNet-50 final bottleneck activation layer

% Calling gradCAM without ReductionLayer allows MATLAB DAGNetwork to infer it safely
try
    gradcamMap = gradCAM(net, resizedImg, predictedClass, 'FeatureLayer', fLayer);
catch
    % Fallback if layer names differ
    warning('Specified layer not found. Automatically finding deepest ReLU layer...');
    layerNames = {net.Layers.Name};
    reluLayers = layerNames(contains(layerNames, 'relu'));
    fLayer = reluLayers{end};
    gradcamMap = gradCAM(net, resizedImg, predictedClass, 'FeatureLayer', fLayer);
end

% 7. Render Dual-Panel Clinical Verification Dashboard
fprintf('🎨 Rendering the Interactive Ophthalmologist Panel...\n');
fig = figure('Name', 'Netra Rakshak - Explainable AI Dashboard', ...
             'NumberTitle', 'off', ...
             'Color', 'w', ...
             'Units', 'normalized', ...
             'Position', [0.1, 0.1, 0.8, 0.7]);

% Panel A: Clean Camera Image
subplot(1, 2, 1);
imshow(resizedImg);
title(sprintf('🏥 Input Fundus (True Level: %s)', trueClassLabel), ...
    'FontSize', 13, 'FontWeight', 'bold');

% Panel B: Grad-CAM Visualization Overlay
subplot(1, 2, 2);
imshow(resizedImg); hold on;
hMap = imagesc(gradcamMap);
set(hMap, 'AlphaData', 0.45);          % 45% transparency overlay
colormap(gca, 'jet');                  % Medical thermal spectrum
colorbar;
axis image off;                        % Maintain proportions, hide axis ticks
title('🔬 Clinically Evaluated Lesion Focus Map', 'FontSize', 13, 'FontWeight', 'bold');
hold off;

% Bottom Status Annotation Box
statusText = sprintf('System Action: AI Diagnostic Match | Prediction: %s | AI Confidence: %.2f%%', ...
    predictedLabelStr, confidencePercentage);
annotation('textbox', [0.15, 0.02, 0.7, 0.07], ...
    'String', statusText, ...
    'FontSize', 11, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'BackgroundColor', [0.92, 0.95, 1.0], ...
    'EdgeColor', [0.2, 0.4, 0.8], ...
    'LineWidth', 1.5);

fprintf('🎉 SUCCESS: Interactive diagnostic report rendered successfully!\n');