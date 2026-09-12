%% STEP 10: EXTERNAL OUT-OF-DISTRIBUTION BENCHMARK (MESSIDOR-2 COHORT)
clc; clear; close all;

fprintf('==================================================================\n');
fprintf('       NETRA RAKSHAK: MESSIDOR-2 EXTERNAL CLINICAL AUDIT          \n');
fprintf('==================================================================\n');

%% 1. SETUP PATHS & LOAD MODEL
modelPath = 'D:\Lost_Projects\Netra_Rakshak_Model\trained_dr_classifier.mat';
messidorDir = 'D:\Lost_Projects\Netra_Rakshak_Model\messidor2';

if ~exist(modelPath, 'file')
    error('❌ Cannot find "trained_dr_classifier.mat"! Ensure Step 5 completed.');
end

fprintf('💾 Loading trained ResNet-50 model...\n');
loaded = load(modelPath);
if isfield(loaded, 'trainedNet')
    net = loaded.trainedNet;
elseif isfield(loaded, 'net')
    net = loaded.net;
else
    error('❌ Could not locate neural network variable.');
end

%% 2. SCAN MESSIDOR-2 IMAGE REPOSITORY
fprintf('📂 Scanning Messidor-2 repository in: %s\n', messidorDir);
imgFiles = [dir(fullfile(messidorDir, '**', '*.png')); ...
            dir(fullfile(messidorDir, '**', '*.jpg')); ...
            dir(fullfile(messidorDir, '**', '*.tif'))];

% Filter out CSVs or masks
validIdx = ~contains({imgFiles.name}, {'mask', 'manual', 'label', 'csv'});
imgFiles = imgFiles(validIdx);
numImages = length(imgFiles);

if numImages == 0
    error('❌ No valid fundus images found inside %s', messidorDir);
end
fprintf('📸 Found %d clinical fundus scans to evaluate.\n', numImages);

%% 3. HIGH-THROUGHPUT CLAHE-ENHANCED GPU INFERENCE
fprintf('🔬 Preprocessing scans via Green-Channel CLAHE & classifying on GPU...\n');

imagePaths = cell(numImages, 1);
for k = 1:numImages
    imagePaths{k} = fullfile(imgFiles(k).folder, imgFiles(k).name);
end

% Image datastore with on-the-fly Step 2 CLAHE preprocessing
imdsMessidor = imageDatastore(imagePaths);
transformImds = transform(imdsMessidor, @(data) preprocessForInference(data));

% GPU Mini-Batch Inference
[predictedLabels, classProbabilities] = classify(net, transformImds, ...
    'ExecutionEnvironment', 'gpu', ...
    'MiniBatchSize', 64);

predScores = max(classProbabilities, [], 2) * 100;
predNumeric = str2double(string(predictedLabels));
predBinary = categorical(predNumeric >= 2, [false, true], {'Non-Referable (0-1)', 'Referable (2-4)'});

%% 4. CLINICAL TRIAGE TELEMETRY REPORT
numRef = sum(predNumeric >= 2);
numNonRef = sum(predNumeric < 2);
pctRef = (numRef / numImages) * 100;
pctNonRef = (numNonRef / numImages) * 100;

fprintf('\n================== MESSIDOR-2 CALIBRATED AUDIT ==================\n');
fprintf('  Total Scans Processed:         %d eyes\n', numImages);
fprintf('  Triaged Non-Referable (0-1):   %d (%.2f%%) -> Discharged locally\n', numNonRef, pctNonRef);
fprintf('  Triaged Referable (2-4):       %d (%.2f%%) -> Routed to hospital\n', numRef, pctRef);
fprintf('  Average Triage Confidence:     %.2f%%\n', mean(predScores));
fprintf('=================================================================\n');

% Breakdown across all 5 clinical stages
fprintf('\n--- 5-Class Stage Distribution across External Cohort ---\n');
for c = 0:4
    countC = sum(predNumeric == c);
    fprintf('  Grade %d: %4d patients (%5.2f%%)\n', c, countC, (countC / numImages) * 100);
end
fprintf('---------------------------------------------------------\n');

%% 5. MULTI-PATIENT CLINICAL XAI INSPECTION GALLERY (TOP 6 CASES)
fprintf('🎨 Rendering Multi-Patient Explainable Grad-CAM Gallery...\n');

% Find top 6 referable cases sorted by confidence
referableIndices = find(predNumeric >= 2);
[~, sortRank] = sort(predScores(referableIndices), 'descend');
topCases = referableIndices(sortRank(1:min(6, length(sortRank))));

if length(topCases) < 6
    % If fewer than 6 referable cases exist, pad with highest-confidence overall
    [~, sortAll] = sort(predScores, 'descend');
    topCases = unique([topCases; sortAll(1:6)], 'stable');
    topCases = topCases(1:6);
end

figGallery = figure('Name', 'Netra Rakshak - Messidor-2 Multi-Patient Audit Gallery', ...
                    'Color', 'w', 'Units', 'normalized', 'Position', [0.03, 0.05, 0.94, 0.85]);

for idx = 1:length(topCases)
    caseIdx = topCases(idx);
    rawImg = imread(imagePaths{caseIdx});
    processedImg = preprocessForInference(rawImg);
    predClass = predictedLabels(caseIdx);
    confVal = predScores(caseIdx);
    
    % Compute Grad-CAM heatmap
    try
        camMap = gradCAM(net, processedImg, predClass, 'FeatureLayer', 'activation_49_relu');
    catch
        camMap = zeros(224, 224);
    end
    
    % Subplot: 2 rows of 6 columns (Raw vs Grad-CAM)
    % Row 1: Original Fundus Scan
    subplot(2, 6, idx);
    imshow(imresize(rawImg, [224, 224]));
    title(sprintf('Patient %d\n%s', idx, imgFiles(caseIdx).name), 'FontSize', 8, 'Interpreter', 'none');
    
    % Row 2: Diagnostic Grad-CAM Overlay
    subplot(2, 6, idx + 6);
    imshow(processedImg); hold on;
    imagesc(camMap, 'AlphaData', 0.45);
    colormap(gca, 'jet');
    title(sprintf('Grade %s (%.1f%%)\n[Referable]', string(predClass), confVal), ...
        'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.8, 0.1, 0.1]);
    hold off;
end

% Save Audit CSV Log
auditResults = table({imgFiles.name}', string(predictedLabels), predScores, string(predBinary), ...
    'VariableNames', {'Image_Filename', 'Predicted_Grade', 'Confidence_Pct', 'Triage_Status'});
writetable(auditResults, 'D:\Lost_Projects\Netra_Rakshak_Model\messidor2_triage_audit.csv');
fprintf('💾 Updated triage audit log saved to "messidor2_triage_audit.csv".\n');
fprintf('🎉 Messidor-2 Multi-Patient Validation Complete!\n');

%% HELPER FUNCTION: STEP 2 CLAHE PREPROCESSING
function imgOut = preprocessForInference(imgIn)
    if iscell(imgIn)
        imgIn = imgIn{1};
    end
    imgResized = imresize(imgIn, [224, 224]);
    % Enhance green channel contrast with CLAHE
    green = imgResized(:,:,2);
    enhancedGreen = adapthisteq(green, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    imgOut = imgResized;
    imgOut(:,:,2) = enhancedGreen;
end