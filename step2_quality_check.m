%% STEP 2: HIGH-PERFORMANCE IMAGE QUALITY ASSESSMENT (IQA) & ENHANCEMENT
clc; clear; close all;

% 1. Define your folder paths
inputBaseDir = 'D:\Lost_Projects\Netra_Rakshak_Model\sorted_dataset\';
outputBaseDir = 'D:\Lost_Projects\Netra_Rakshak_Model\enhanced_dataset\';

fprintf('Initializing High-Performance IQA Pipeline...\n');

% 2. UNLEASH THE i9 PROCESSOR: Start the Parallel Computing Pool
% This opens up "workers" across your CPU cores to process images simultaneously.
if isempty(gcp('nocreate'))
    fprintf('Starting CPU Parallel Pool... Please wait a moment.\n');
    parpool; 
end

% 3. Loop through each of the 5 severity folders (0 to 4)
for severity = 0:4
    currentInFolder = fullfile(inputBaseDir, num2str(severity));
    currentOutFolder = fullfile(outputBaseDir, num2str(severity));
    
    if ~exist(currentOutFolder, 'dir')
        mkdir(currentOutFolder);
    end
    
    imgFiles = dir(fullfile(currentInFolder, '*.png'));
    numFiles = length(imgFiles);
    
    if numFiles == 0
        continue; 
    end
    
    fprintf('\n🚀 Processing Severity Level %d (%d images found)...\n', severity, numFiles);
    
    % 4. VISUAL PROGRESS BAR
    % This prevents MATLAB from looking "stuck" by showing a graphical loading bar.
    progressBar = waitbar(0, sprintf('Processing Severity Level %d...', severity), 'Name', 'Netra Rakshak Pipeline');
    
    % We use 'parfor' instead of 'for' to split the 1,805 images across your CPU cores!
    parfor i = 1:numFiles
        imgPath = fullfile(currentInFolder, imgFiles(i).name);
        rawImg = imread(imgPath);
        
        % --- PHASE 1: IMAGE QUALITY ASSESSMENT ---
        grayImg = rgb2gray(rawImg);
        
        % Focus Check (Laplacian Edge Sharpness)
        lapFilter = [0 1 0; 1 -4 1; 0 1 0];
        laplacian = imfilter(double(grayImg), lapFilter, 'replicate');
        blurScore = var(laplacian(:));
        
        % Illumination Check
        illuminationScore = std(double(grayImg(:)));
        
        % Quality Gate Check: Skip unusable images
        if blurScore < 8 || illuminationScore < 15
            continue; 
        end
        
        % --- PHASE 2: ADAPTIVE ENHANCEMENT ---
        greenChannel = rawImg(:,:,2);
        
        % Apply CLAHE
        enhancedGreen = adapthisteq(greenChannel, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
        
        % Denoising
        denoisedGreen = medfilt2(enhancedGreen, [3 3]);
        
        % Reconstruct
        enhancedImg = rawImg;
        enhancedImg(:,:,2) = denoisedGreen;
        
        % Save to disk
        outPath = fullfile(currentOutFolder, imgFiles(i).name);
        imwrite(enhancedImg, outPath);
    end
    
    % Close the progress bar window for this severity folder
    if ishandle(progressBar)
        close(progressBar);
    end
end

fprintf('\n🎉 SUCCESS: Quality Check and Enhancement Complete!\n');
fprintf('Perfect training data saved in: "D:\\Lost_Projects\\Netra_Rakshak_Model\\enhanced_dataset\\"\n');
