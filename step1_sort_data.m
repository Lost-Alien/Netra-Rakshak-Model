%% STEP 1: AUTOMATED DATASET SORTER FOR NETRA RAKSHAK MODEL
clc; clear; close all;

% 1. Explicit folder paths mapped to your D drive setup
csvFile = 'D:\Lost_Projects\Netra_Rakshak_Model\1_APTOS_Data\train.csv';
sourceImageDir = 'D:\Lost_Projects\Netra_Rakshak_Model\1_APTOS_Data\train_images\';
outputBaseDir = 'D:\Lost_Projects\Netra_Rakshak_Model\sorted_dataset\';

% 2. Read the CSV file containing labels
fprintf('Reading APTOS CSV mapping data...\n');
if ~exist(csvFile, 'file')
    error('Could not find train.csv! Please verify your folder path configuration.');
end
dataTable = readtable(csvFile);

% 3. Create target subfolders (0 to 4) if they don't exist yet
for i = 0:4
    targetFolder = fullfile(outputBaseDir, num2str(i));
    if ~exist(targetFolder, 'dir')
        mkdir(targetFolder);
    end
end

% 4. Loop through the table and copy files to their respective severity folders
totalImages = height(dataTable);
fprintf('Starting image sorting. Total images to process: %d\n', totalImages);

for row = 1:totalImages
    % Extract image name and its specific diagnosis level
    imgName = dataTable.id_code{row}; 
    severity = dataTable.diagnosis(row);

    % Build full file paths
    sourceFile = fullfile(sourceImageDir, [imgName '.png']);
    destinationFile = fullfile(outputBaseDir, num2str(severity), [imgName '.png']);

    % Check if the image file actually exists before copying
    if exist(sourceFile, 'file')
        copyfile(sourceFile, destinationFile);
    else
        % Some datasets use .jpeg or .jpg instead of .png - let's check for those too
        altSourceFile = fullfile(sourceImageDir, [imgName '.jpeg']);
        if exist(altSourceFile, 'file')
            destinationFileAlt = fullfile(outputBaseDir, num2str(severity), [imgName '.jpeg']);
            copyfile(altSourceFile, destinationFileAlt);
        end
    end

    % Display progress updates every 500 images processed
    if mod(row, 500) == 0
        fprintf('Successfully sorted %d / %d images...\n', row, totalImages);
    end
end

fprintf('🎉 Dataset sorting complete! Look inside your "D:\\Lost_Projects\\Netra_Rakshak_Model\\sorted_dataset\\" folder.\n');
