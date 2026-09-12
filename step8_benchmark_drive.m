%% STEP 8: QUANTITATIVE BENCHMARK AGAINST DRIVE DATASET (VESSEL SEGMENTATION)
clc; clear; close all;

% 1. Setup DRIVE Dataset Path
% Pointing directly to the cohort folder containing images 21-40 and 1st_manual ground truth
driveBaseDir = 'D:\Lost_Projects\XAI dr\data\drive\training\training';

if ~exist(driveBaseDir, 'dir')
    % Fallback if paths were rearranged
    driveBaseDir = 'D:\Lost_Projects\New folder (2)\XAI dr\data\drive\training\training';
end

if ~exist(driveBaseDir, 'dir')
    error('❌ Cannot find DRIVE directory! Please verify folder path.');
end

% Locate fundus images (*.tif)
allTifs = dir(fullfile(driveBaseDir, '**', '*.tif'));
imgSearch = allTifs(~contains({allTifs.name}, {'manual', 'mask'}));

if isempty(imgSearch)
    error('❌ Could not locate .tif fundus images in: %s', driveBaseDir);
end

numImages = length(imgSearch);
fprintf('📂 Connected to DRIVE cohort at: %s\n', driveBaseDir);
fprintf('📸 Evaluating %d paired retinal scans against expert ground truth...\n', numImages);

% 2. Initialize Metric Accumulators
diceScores    = zeros(numImages, 1);
jaccardScores = zeros(numImages, 1);
sensScores    = zeros(numImages, 1);
specScores    = zeros(numImages, 1);
accScores     = zeros(numImages, 1);
imageNames    = cell(numImages, 1);

% Prepare figure for visual qualitative verification (First 4 samples)
figQual = figure('Name', 'Netra Rakshak - DRIVE Segmentation Benchmark', ...
                 'Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.85]);

fprintf('\n🔬 Running Multiscale Directional Morphology across DRIVE cohort...\n');
fprintf('----------------------------------------------------------------------\n');
fprintf(' Img ID |   Dice (F1)  |  Jaccard (IoU) | Sensitivity | Specificity | Accuracy \n');
fprintf('----------------------------------------------------------------------\n');

for i = 1:numImages
    % 3. Read Raw Fundus Image
    imgPath = fullfile(imgSearch(i).folder, imgSearch(i).name);
    rawRGB = imread(imgPath);
    [origH, origW, ~] = size(rawRGB);
    
    [~, baseName, ~] = fileparts(imgSearch(i).name);
    % Extract numeric prefix (e.g., '21', '22')
    numTokens = regexp(baseName, '^\d+', 'match');
    if isempty(numTokens)
        numTokens = regexp(baseName, '\d+', 'match');
    end
    prefix = numTokens{1};
    imageNames{i} = prefix;
    
    % 4. Load Matching 1st_manual Ground Truth Mask
    gtFiles = dir(fullfile(driveBaseDir, '**', [prefix '_manual1.*']));
    if isempty(gtFiles)
        gtFiles = dir(fullfile(driveBaseDir, '**', ['*' prefix '*manual*']));
    end
    
    if isempty(gtFiles)
        error('❌ Missing ground truth annotation for image %s', prefix);
    end
    
    gtRaw = imread(fullfile(gtFiles(1).folder, gtFiles(1).name));
    gtBinary = gtRaw > 0;
    if size(gtBinary, 3) > 1
        gtBinary = rgb2gray(gtRaw) > 0;
    end
    
    % 5. Load Matching FoV Mask
    maskFiles = dir(fullfile(driveBaseDir, '**', [prefix '*mask*']));
    if ~isempty(maskFiles)
        maskRaw = imread(fullfile(maskFiles(1).folder, maskFiles(1).name));
        fovMask = maskRaw > 0;
        if size(fovMask, 3) > 1
            fovMask = rgb2gray(maskRaw) > 0;
        end
    else
        % Fallback: Compute FoV boundary from luminance
        grayImg = rgb2gray(rawRGB);
        fovMask = imbinarize(grayImg, 0.05);
        fovMask = imfill(fovMask, 'holes');
        fovMask = bwareafilt(fovMask, 1);
        fovMask = imerode(fovMask, strel('disk', 8));
    end
    
    % 6. Multi-Angle Linear Morphology Vessel Extraction
    greenChan = im2double(rawRGB(:,:,2));
    invGreen = 1 - greenChan;
    
    % Local illumination leveling
    bgIllum = imopen(invGreen, strel('disk', 15));
    vesselsEnhanced = max(0, invGreen - bgIllum);
    
    % Multi-angle sweeps (12 radial orientations)
    vesselMaxResp = zeros(origH, origW);
    for angle = 0:15:165
        seLine = strel('line', 11, angle);
        opened = imopen(vesselsEnhanced, seLine);
        vesselMaxResp = max(vesselMaxResp, opened);
    end
    vesselMaxResp = vesselMaxResp .* fovMask;
    
    % Adaptive vessel thresholding
    vesselCutoff = prctile(vesselMaxResp(fovMask), 87);
    predVessels = (vesselMaxResp > vesselCutoff) & fovMask;
    predVessels = bwareaopen(predVessels, 20);
    predVessels = imclose(predVessels, strel('disk', 1)) & fovMask;
    
    % 7. Compute Metrics (Strictly Inside Field of View)
    evalPixels = fovMask(:);
    yTrue = gtBinary(:);
    yPred = predVessels(:);
    
    yTrueFov = yTrue(evalPixels);
    yPredFov = yPred(evalPixels);
    
    TP = sum(yTrueFov & yPredFov);
    FP = sum(~yTrueFov & yPredFov);
    FN = sum(yTrueFov & ~yPredFov);
    TN = sum(~yTrueFov & ~yPredFov);
    
    diceScores(i)    = (2 * TP) / (2 * TP + FP + FN);
    jaccardScores(i) = TP / (TP + FP + FN);
    sensScores(i)    = TP / (TP + FN);
    specScores(i)    = TN / (TN + FP);
    accScores(i)     = (TP + TN) / (TP + TN + FP + FN);
    
    fprintf('   %s   |    %.4f    |     %.4f     |   %.4f    |   %.4f    |  %.4f\n', ...
        prefix, diceScores(i), jaccardScores(i), sensScores(i), specScores(i), accScores(i));
    
    % 8. Render Visual Proof Panel for the First 4 Cases
    if i <= 4
        subplot(4, 4, (i-1)*4 + 1);
        imshow(rawRGB);
        title(sprintf('Image %s: Fundus', prefix), 'FontSize', 9, 'FontWeight', 'bold');
        
        subplot(4, 4, (i-1)*4 + 2);
        imshow(gtBinary);
        title('Ground Truth (Manual)', 'FontSize', 9, 'FontWeight', 'bold');
        
        subplot(4, 4, (i-1)*4 + 3);
        imshow(predVessels);
        title(sprintf('Netra Rakshak (Dice: %.2f)', diceScores(i)), 'FontSize', 9, 'FontWeight', 'bold');
        
        subplot(4, 4, (i-1)*4 + 4);
        errorMap = zeros(origH, origW, 3);
        errorMap(:,:,2) = (predVessels & gtBinary);  % Green = True Positives
        errorMap(:,:,1) = (predVessels & ~gtBinary); % Red = Over-segmentation
        errorMap(:,:,3) = (~predVessels & gtBinary); % Blue = Missed fine branches
        imshow(errorMap);
        title('Error (G:TP, R:FP, B:FN)', 'FontSize', 9, 'FontWeight', 'bold');
    end
end

% 9. Final Benchmark Report
fprintf('----------------------------------------------------------------------\n');
fprintf('  MEAN SCORE (DRIVE BENCHMARK, N=%d):\n', numImages);
fprintf('  Dice Coefficient (F1):    %.4f  ± %.4f\n', mean(diceScores), std(diceScores));
fprintf('  Jaccard Index (IoU):      %.4f  ± %.4f\n', mean(jaccardScores), std(jaccardScores));
fprintf('  Vessel Sensitivity:       %.4f  ± %.4f\n', mean(sensScores), std(sensScores));
fprintf('  Vessel Specificity:       %.4f  ± %.4f\n', mean(specScores), std(specScores));
fprintf('  Global Pixel Accuracy:    %.4f  ± %.4f\n', mean(accScores), std(accScores));
fprintf('======================================================================\n');
fprintf('🎉 DRIVE Quantitative Benchmark Complete!\n');