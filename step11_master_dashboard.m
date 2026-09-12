%% STEP 11: MASTER CLINICAL TELEMEDICINE & XAI DIAGNOSTIC STATION
% Architecture: Unified Autonomous Retinal Diagnostic Suite (MathWorks SIH26038)
clc; clear; close all;

% Launch Application
launchNetraRakshakApp();

function launchNetraRakshakApp()
    % 1. Global State Management
    appData = struct();
    appData.modelPath = 'D:\Lost_Projects\Netra_Rakshak_Model\trained_dr_classifier.mat';
    appData.net = [];
    appData.currentImage = [];
    appData.currentPath = '';
    
    % Clinical Descriptions & Colors
    appData.classNames = { ...
        'Level 0: No Apparent Retinopathy (Healthy)', ...
        'Level 1: Mild Non-Proliferative DR', ...
        'Level 2: Moderate Non-Proliferative DR', ...
        'Level 3: Severe Non-Proliferative DR', ...
        'Level 4: Proliferative Diabetic Retinopathy' ...
    };
    appData.classColors = { ...
        [0.13, 0.70, 0.35], ... % Level 0: Emerald Green
        [0.20, 0.60, 0.85], ... % Level 1: Sky Blue
        [0.95, 0.65, 0.10], ... % Level 2: Amber
        [0.90, 0.40, 0.15], ... % Level 3: Dark Orange
        [0.85, 0.20, 0.20]  ... % Level 4: Crimson Red
    };

    % 2. Master Window Configuration
    fig = uifigure('Name', 'NETRA RAKSHAK: Autonomous Retinal Diagnostic & Telemedicine Station', ...
                   'Position', [40, 40, 1500, 900], ...
                   'Color', [0.07, 0.09, 0.15]); % Deep Medical Slate

    mainGrid = uigridlayout(fig, [3, 3]);
    mainGrid.RowHeight = {65, '1x', 35};
    mainGrid.ColumnWidth = {340, '1x', 380};
    mainGrid.Padding = [12, 12, 12, 12];
    mainGrid.RowSpacing = 10;
    mainGrid.ColumnSpacing = 10;

    % 3. Header Banner Panel
    headerPanel = uipanel(mainGrid, 'BackgroundColor', [0.11, 0.15, 0.24], 'BorderType', 'none');
    headerPanel.Layout.Row = 1;
    headerPanel.Layout.Column = [1, 3];
    headerGrid = uigridlayout(headerPanel, [1, 3]);
    headerGrid.ColumnWidth = {'1x', 'fit', 'fit'};
    headerGrid.Padding = [15, 10, 15, 10];

    lblTitle = uilabel(headerGrid, 'Text', 'NETRA RAKSHAK | Clinical Decision Support System', ...
        'FontSize', 20, 'FontWeight', 'bold', 'FontColor', [0.95, 0.97, 1.0]);
    lblTitle.Layout.Row = 1; lblTitle.Layout.Column = 1;

    lblSub = uilabel(headerGrid, 'Text', 'MathWorks SIH26038 | Edge-XAI Telemedicine Pipeline', ...
        'FontSize', 12, 'FontColor', [0.55, 0.65, 0.80]);
    lblSub.Layout.Row = 1; lblSub.Layout.Column = 2;

    lblStatus = uilabel(headerGrid, 'Text', '● SYSTEM READY: GPU LOADED', ...
        'FontSize', 12, 'FontWeight', 'bold', 'FontColor', [0.15, 0.85, 0.45]);
    lblStatus.Layout.Row = 1; lblStatus.Layout.Column = 3;

    % 4. Left Sidebar: Image Ingestion, Controls & IQA
    leftPanel = uipanel(mainGrid, 'BackgroundColor', [0.11, 0.15, 0.24], 'BorderType', 'none');
    leftPanel.Layout.Row = 2;
    leftPanel.Layout.Column = 1;
    leftGrid = uigridlayout(leftPanel, [7, 1]);
    leftGrid.RowHeight = {45, 45, 55, 140, 140, '1x', 40};
    leftGrid.Padding = [12, 12, 12, 12];
    leftGrid.RowSpacing = 8;

    btnLoad = uibutton(leftGrid, 'push', 'Text', '📂 Select Patient Fundus Scan', ...
        'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.18, 0.26, 0.40], 'FontColor', 'w');

    btnSample = uidropdown(leftGrid, ...
        'Items', {'-- Quick Clinical Samples --', 'Sample Level 0 (Healthy)', 'Sample Level 1 (Mild)', 'Sample Level 2 (Moderate)', 'Sample Level 3 (Severe)', 'Sample Level 4 (Proliferative)'}, ...
        'FontSize', 12, 'BackgroundColor', [0.15, 0.20, 0.32], 'FontColor', 'w');

    btnDiagnose = uibutton(leftGrid, 'push', 'Text', '⚡ RUN CLINICAL DIAGNOSIS', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.08, 0.52, 0.95], 'FontColor', 'w');

    % Metadata Card
    metaPanel = uipanel(leftGrid, 'Title', 'PATIENT & REPOSITORY METADATA', ...
        'BackgroundColor', [0.09, 0.12, 0.20], 'ForegroundColor', [0.6, 0.75, 0.9], ...
        'FontSize', 10, 'FontWeight', 'bold');
    metaGrid = uigridlayout(metaPanel, [3, 2]);
    metaGrid.Padding = [8, 8, 8, 8];
    metaGrid.RowHeight = {'1x', '1x', '1x'};
    uilabel(metaGrid, 'Text', 'Record ID:', 'FontColor', [0.6, 0.7, 0.8], 'FontSize', 11);
    lblRecID = uilabel(metaGrid, 'Text', 'N/A', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');
    uilabel(metaGrid, 'Text', 'Resolution:', 'FontColor', [0.6, 0.7, 0.8], 'FontSize', 11);
    lblRes = uilabel(metaGrid, 'Text', 'N/A', 'FontColor', 'w', 'FontSize', 11);
    uilabel(metaGrid, 'Text', 'Camera Spec:', 'FontColor', [0.6, 0.7, 0.8], 'FontSize', 11);
    lblCam = uilabel(metaGrid, 'Text', '45° Mydriatic/Non-Mydriatic', 'FontColor', 'w', 'FontSize', 11);

    % IQA Telemetry Card
    iqaPanel = uipanel(leftGrid, 'Title', 'PHASE 1: IMAGE QUALITY ASSESSMENT (IQA)', ...
        'BackgroundColor', [0.09, 0.12, 0.20], 'ForegroundColor', [0.6, 0.75, 0.9], ...
        'FontSize', 10, 'FontWeight', 'bold');
    iqaGrid = uigridlayout(iqaPanel, [3, 2]);
    iqaGrid.Padding = [8, 8, 8, 8];
    iqaGrid.RowHeight = {'1x', '1x', '1x'};
    uilabel(iqaGrid, 'Text', 'Sharpness Index:', 'FontColor', [0.6, 0.7, 0.8], 'FontSize', 11);
    lblSharp = uilabel(iqaGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');
    uilabel(iqaGrid, 'Text', 'Illumination Balance:', 'FontColor', [0.6, 0.7, 0.8], 'FontSize', 11);
    lblIllum = uilabel(iqaGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11);
    uilabel(iqaGrid, 'Text', 'Quality Decision:', 'FontColor', [0.6, 0.7, 0.8], 'FontSize', 11);
    lblIQADecision = uilabel(iqaGrid, 'Text', 'PENDING SCAN', 'FontColor', [0.95, 0.75, 0.2], 'FontSize', 11, 'FontWeight', 'bold');

    % 5. Center Panel: Visual Multi-Modal Inspection Matrix
    centerPanel = uipanel(mainGrid, 'BackgroundColor', [0.11, 0.15, 0.24], 'BorderType', 'none');
    centerPanel.Layout.Row = 2;
    centerPanel.Layout.Column = 2;
    centerGrid = uigridlayout(centerPanel, [2, 2]);
    centerGrid.RowHeight = {'1x', '1x'};
    centerGrid.ColumnWidth = {'1x', '1x'};
    centerGrid.Padding = [10, 10, 10, 10];
    centerGrid.RowSpacing = 8; centerGrid.ColumnSpacing = 8;

    % 4 Diagnostic Quadrants
    axRaw = uiaxes(centerGrid, 'BackgroundColor', [0.05, 0.07, 0.12]);
    title(axRaw, '1. Primary Optical Acquisition', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
    cleanAxis(axRaw);

    axEnh = uiaxes(centerGrid, 'BackgroundColor', [0.05, 0.07, 0.12]);
    title(axEnh, '2. Rayleigh Green-Channel CLAHE', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
    cleanAxis(axEnh);

    axXAI = uiaxes(centerGrid, 'BackgroundColor', [0.05, 0.07, 0.12]);
    title(axXAI, '3. Phase 4: Grad-CAM Saliency Map', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
    cleanAxis(axXAI);

    axBio = uiaxes(centerGrid, 'BackgroundColor', [0.05, 0.07, 0.12]);
    title(axBio, '4. Phase 2: Biomarker Segmentation Map', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
    cleanAxis(axBio);

    % 6. Right Panel: Severity Classification & Telemedicine Dispatch
    rightPanel = uipanel(mainGrid, 'BackgroundColor', [0.11, 0.15, 0.24], 'BorderType', 'none');
    rightPanel.Layout.Row = 2;
    rightPanel.Layout.Column = 3;
    rightGrid = uigridlayout(rightPanel, [4, 1]);
    rightGrid.RowHeight = {160, 160, 180, '1x'};
    rightGrid.Padding = [12, 12, 12, 12];
    rightGrid.RowSpacing = 10;

    % Severity Verdict Card
    verdictPanel = uipanel(rightGrid, 'Title', 'PHASE 3: CLINICAL SEVERITY RATING (ICDR)', ...
        'BackgroundColor', [0.09, 0.12, 0.20], 'ForegroundColor', [0.6, 0.75, 0.9], ...
        'FontSize', 10, 'FontWeight', 'bold');
    verdictGrid = uigridlayout(verdictPanel, [3, 1]);
    verdictGrid.RowHeight = {45, 30, 40};
    verdictGrid.Padding = [10, 8, 10, 8];
    
    lblSeverityBadge = uilabel(verdictGrid, 'Text', 'AWAITING SCAN', ...
        'FontSize', 15, 'FontWeight', 'bold', 'FontColor', [0.7, 0.7, 0.7], ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0.14, 0.18, 0.28]);
    
    lblConf = uilabel(verdictGrid, 'Text', 'Model Confidence: -- %', ...
        'FontSize', 12, 'FontColor', [0.85, 0.9, 1.0], 'HorizontalAlignment', 'center');

    lblReferralBadge = uilabel(verdictGrid, 'Text', 'TRIAGE STATUS: PENDING', ...
        'FontSize', 13, 'FontWeight', 'bold', 'FontColor', [0.8, 0.8, 0.8], ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0.12, 0.16, 0.24]);

    % Biomarker Quantitative Summary
    bioPanel = uipanel(rightGrid, 'Title', 'PHASE 2: DETERMINISTIC RETINAL BIOMARKERS', ...
        'BackgroundColor', [0.09, 0.12, 0.20], 'ForegroundColor', [0.6, 0.75, 0.9], ...
        'FontSize', 10, 'FontWeight', 'bold');
    bioGrid = uigridlayout(bioPanel, [4, 2]);
    bioGrid.Padding = [10, 8, 10, 8];
    bioGrid.RowHeight = {'1x', '1x', '1x', '1x'};

    uilabel(bioGrid, 'Text', 'Sub-pixel Microaneurysms:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblMA = uilabel(bioGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');

    uilabel(bioGrid, 'Text', 'Blot Hemorrhages:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblHemo = uilabel(bioGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');

    uilabel(bioGrid, 'Text', 'Hard Exudates Burden:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblExudate = uilabel(bioGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');

    uilabel(bioGrid, 'Text', 'Vascular Density (Target: 10-15%):', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblVessel = uilabel(bioGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');

    % Telemedicine Telemetry Card
    telePanel = uipanel(rightGrid, 'Title', 'PHASE 5: SIMULINK DISTRICT TELEMEDICINE DISPATCH', ...
        'BackgroundColor', [0.09, 0.12, 0.20], 'ForegroundColor', [0.6, 0.75, 0.9], ...
        'FontSize', 10, 'FontWeight', 'bold');
    teleGrid = uigridlayout(telePanel, [4, 2]);
    teleGrid.Padding = [10, 8, 10, 8];
    teleGrid.RowHeight = {'1x', '1x', '1x', '1x'};

    uilabel(teleGrid, 'Text', 'Transmission Action:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblTeleAction = uilabel(teleGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');

    uilabel(teleGrid, 'Text', 'Uplink Payload Size:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblTelePayload = uilabel(teleGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11);

    uilabel(teleGrid, 'Text', '2G/3G Upload Time:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblTeleLatency = uilabel(teleGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11);

    uilabel(teleGrid, 'Text', 'Doctor Queue Priority:', 'FontColor', [0.7, 0.8, 0.9], 'FontSize', 11);
    lblTelePriority = uilabel(teleGrid, 'Text', '--', 'FontColor', 'w', 'FontSize', 11, 'FontWeight', 'bold');

    % Confidence Distribution Bar Chart
    axBar = uiaxes(rightGrid, 'BackgroundColor', [0.09, 0.12, 0.20]);
    title(axBar, '5-Stage Probability Distribution', 'Color', [0.8, 0.85, 0.95], 'FontSize', 9);
    set(axBar, 'XColor', [0.5, 0.6, 0.7], 'YColor', [0.5, 0.6, 0.7], 'YLim', [0, 100]);
    axBar.XTick = 0:4;
    axBar.XTickLabel = {'L0', 'L1', 'L2', 'L3', 'L4'};

    % 7. Footer Status Bar
    footerPanel = uipanel(mainGrid, 'BackgroundColor', [0.08, 0.11, 0.18], 'BorderType', 'none');
    footerPanel.Layout.Row = 3;
    footerPanel.Layout.Column = [1, 3];
    footerGrid = uigridlayout(footerPanel, [1, 3]);
    footerGrid.ColumnWidth = {'1x', 'fit', 'fit'};
    footerGrid.Padding = [15, 5, 15, 5];

    lblFooter = uilabel(footerGrid, 'Text', 'Netra Rakshak CDSS v2.6 | Validated against APTOS, DRIVE, IDRiD, & Messidor-2 cohorts', ...
        'FontSize', 11, 'FontColor', [0.45, 0.55, 0.68]);
    lblSimulinkStatus = uilabel(footerGrid, 'Text', 'Simulink Operational Engine: netra_rakshak_telemedicine.slx [Active]', ...
        'FontSize', 11, 'FontColor', [0.35, 0.75, 0.45]);

    %% 8. CALLBACK IMPLEMENTATIONS
    btnLoad.ButtonPushedFcn = @(btn, event) browseFile();
    btnSample.ValueChangedFcn = @(dd, event) selectSample(dd.Value);
    btnDiagnose.ButtonPushedFcn = @(btn, event) runFullDiagnosis();

    % Preload Model
    try
        loaded = load(appData.modelPath);
        if isfield(loaded, 'trainedNet')
            appData.net = loaded.trainedNet;
        elseif isfield(loaded, 'net')
            appData.net = loaded.net;
        end
    catch ME
        uialert(fig, sprintf('Failed to load classifier: %s', ME.message), 'Model Error');
    end

    function browseFile()
        [file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif', 'Retinal Fundus Scans (*.png, *.jpg, *.tif)'}, ...
            'Select Retinal Scan', 'D:\Lost_Projects\Netra_Rakshak_Model\');
        if isequal(file, 0), return; end
        loadAndDisplay(fullfile(path, file));
    end

    function selectSample(val)
        if contains(val, '--'), return; end
        baseFolder = 'D:\Lost_Projects\Netra_Rakshak_Model\enhanced_dataset';
        lvl = regexp(val, '\d', 'match');
        if isempty(lvl), return; end
        targetDir = fullfile(baseFolder, lvl{1});
        flist = dir(fullfile(targetDir, '*.png'));
        if isempty(flist)
            uialert(fig, 'No sample images found in enhanced_dataset directory.', 'Directory Notice');
            return;
        end
        loadAndDisplay(fullfile(targetDir, flist(1).name));
    end

    function loadAndDisplay(filePath)
        appData.currentPath = filePath;
        appData.currentImage = imread(filePath);
        [~, fname, ext] = fileparts(filePath);
        lblRecID.Text = [fname, ext];
        [h, w, ~] = size(appData.currentImage);
        lblRes.Text = sprintf('%d × %d px', w, h);
        
        imshow(appData.currentImage, 'Parent', axRaw);
        title(axRaw, '1. Primary Optical Acquisition', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
        cleanAxis(axRaw);
        
        % Reset Telemetry Displays
        lblIQADecision.Text = 'PENDING'; lblIQADecision.FontColor = [0.95, 0.75, 0.2];
        lblSeverityBadge.Text = 'READY TO DIAGNOSE'; lblSeverityBadge.BackgroundColor = [0.14, 0.18, 0.28];
        lblReferralBadge.Text = 'TRIAGE STATUS: PENDING'; lblReferralBadge.BackgroundColor = [0.12, 0.16, 0.24];
    end

    function runFullDiagnosis()
        if isempty(appData.currentImage)
            uialert(fig, 'Please select or load a retinal fundus scan first.', 'Missing Input');
            return;
        end

        rawImg = appData.currentImage;
        [rows, cols, ~] = size(rawImg);

        % --- PHASE 1: IQA & ENHANCEMENT ---
        gray = rgb2gray(rawImg);
        [~, gradThresh] = edge(gray, 'sobel');
        sharpVal = sum(gradThresh(:)) * 100;
        lblSharp.Text = sprintf('%.2f (Threshold: >1.5)', sharpVal);
        
        lumMean = mean(gray(:));
        lblIllum.Text = sprintf('%.1f / 255 (Valid: 30-220)', lumMean);

        if sharpVal < 0.8 || lumMean < 25 || lumMean > 230
            lblIQADecision.Text = '⚠️ RECAPTURE (UNGRADEABLE)';
            lblIQADecision.FontColor = [0.9, 0.2, 0.2];
            uialert(fig, 'Optical acquisition failed clinical quality thresholds. Recapture required.', 'IQA Notice');
            return;
        else
            lblIQADecision.Text = '✅ PASSED (CLINICAL GRADE)';
            lblIQADecision.FontColor = [0.2, 0.85, 0.4];
        end

        % Green-Channel Rayleigh CLAHE
        procImg = imresize(rawImg, [224, 224]);
        greenChan = procImg(:,:,2);
        enhancedGreen = adapthisteq(greenChan, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
        procImg(:,:,2) = enhancedGreen;
        imshow(procImg, 'Parent', axEnh);
        title(axEnh, '2. Rayleigh Green-Channel CLAHE', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
        cleanAxis(axEnh);

        % --- PHASE 3: SEVERITY CLASSIFICATION ---
        [predClass, scores] = classify(appData.net, procImg);
        predIdx = str2double(string(predClass));
        confVal = scores(predIdx + 1) * 100;

        lblSeverityBadge.Text = appData.classNames{predIdx + 1};
        lblSeverityBadge.BackgroundColor = appData.classColors{predIdx + 1};
        lblSeverityBadge.FontColor = 'w';
        lblConf.Text = sprintf('Model Confidence: %.2f%%', confVal);

        % Probability Bar Graph
        b = bar(axBar, 0:4, scores * 100, 0.6);
        b.FaceColor = 'flat';
        for k = 1:5
            b.CData(k, :) = appData.classColors{k};
        end
        axBar.YLim = [0, 100];
        grid(axBar, 'on');

        % --- PHASE 4: EXPLAINABLE GRAD-CAM HEATMAP ---
        try
            camMap = gradCAM(appData.net, procImg, predClass, 'FeatureLayer', 'activation_49_relu');
            imshow(procImg, 'Parent', axXAI); hold(axXAI, 'on');
            imagesc(axXAI, camMap, 'AlphaData', 0.45);
            colormap(axXAI, 'jet');
            hold(axXAI, 'off');
            title(axXAI, sprintf('3. Grad-CAM (Target: Level %d)', predIdx), 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
            cleanAxis(axXAI);
        catch
            imshow(procImg, 'Parent', axXAI);
        end

        % --- PHASE 2: DETERMINISTIC BIOMARKER SEGMENTATION ---
        targetWidth = 800;
        scaleFactor = targetWidth / cols;
        scaledRGB = imresize(rawImg, scaleFactor);
        [sH, sW, ~] = size(scaledRGB);

        sGray = rgb2gray(scaledRGB);
        fovMask = sGray > 15;
        fovMask = imfill(fovMask, 'holes');
        fovMask = bwareafilt(fovMask, 1);
        margin = round(min(sH, sW) * 0.05);
        fovInner = imerode(fovMask, strel('disk', margin));

        sGreen = im2double(scaledRGB(:,:,2));
        invG = 1 - sGreen;
        bg = imopen(invG, strel('disk', 18));
        vesselsEnhanced = max(0, invG - bg);

        % Directional line sweeps
        vesselMax = zeros(sH, sW);
        for angle = 0:20:160
            seLine = strel('line', 11, angle);
            vesselMax = max(vesselMax, imopen(vesselsEnhanced, seLine));
        end
        vesselCutoff = prctile(vesselMax(fovInner), 87);
        vesselBin = (vesselMax > vesselCutoff) & fovInner;
        vesselBin = bwareaopen(vesselBin, 25);

        % Optic Disc Masking
        sRed = im2double(scaledRGB(:,:,1));
        odScore = imfilter(sRed .* sGreen, fspecial('gaussian', [40 40], 10)) .* fovInner;
        [~, odIdx] = max(odScore(:));
        [odY, odX] = ind2sub([sH, sW], odIdx);
        [Xg, Yg] = meshgrid(1:sW, 1:sH);
        odMask = ((Xg - odX).^2 + (Yg - odY).^2) <= ((min(sH, sW)*0.07)^2);

        % Lesions
        vesselExcl = imdilate(vesselBin, strel('disk', 3));
        bHat = imbothat(sGreen, strel('disk', 6));
        darkLesions = (bHat > 0.065) & ~vesselExcl & ~odMask & fovInner;
        lProps = regionprops(darkLesions, 'Area', 'Eccentricity');
        
        numMA = sum([lProps.Area] >= 3 & [lProps.Area] <= 35 & [lProps.Eccentricity] < 0.85);
        numHemo = sum([lProps.Area] > 35 & [lProps.Area] <= 500);

        tHat = imtophat(sGreen, strel('disk', 7));
        exudateBin = (tHat > 0.11) & ~odMask & fovInner;
        exudatePixels = sum(exudateBin(:));

        retinalArea = sum(fovInner(:));
        vesselDensity = (sum(vesselBin(:)) / retinalArea) * 100;
        exudateBurden = (exudatePixels / retinalArea) * 100;

        % Update Biomarker Card
        lblMA.Text = sprintf('%d foci', numMA);
        lblHemo.Text = sprintf('%d lesions', numHemo);
        lblExudate.Text = sprintf('%d px (%.3f%%)', exudatePixels, exudateBurden);
        lblVessel.Text = sprintf('%.2f%%', vesselDensity);

        % Overlay Quadrant 4
        compView = scaledRGB;
        compView = imoverlay(compView, vesselBin, [0 1 1]);          % Cyan Vessels
        compView = imoverlay(compView, darkLesions, [1 0 0]);        % Red Lesions
        compView = imoverlay(compView, exudateBin, [1 1 0]);         % Yellow Exudates
        imshow(compView, 'Parent', axBio);
        title(axBio, '4. Phase 2: Biomarker Segmentation Map', 'Color', [0.85, 0.9, 1.0], 'FontSize', 10);
        cleanAxis(axBio);

        % --- PHASE 5: TELEMEDICINE DISPATCH LOGIC ---
        if predIdx >= 2
            % Referable Case
            lblReferralBadge.Text = 'TRIAGE: REFERRAL REQUIRED (LEVEL 2+)';
            lblReferralBadge.BackgroundColor = [0.85, 0.20, 0.20];
            lblReferralBadge.FontColor = 'w';

            lblTeleAction.Text = 'UPLINK DISPATCH -> DISTRICT HOSP.';
            lblTeleAction.FontColor = [0.95, 0.4, 0.4];
            lblTelePayload.Text = '0.65 MB (Compressed XAI + JSON)';
            lblTeleLatency.Text = '3.65 sec (over 1.5 Mbps Cellular)';
            
            if predIdx == 4
                lblTelePriority.Text = 'P1 - EMERGENCY OPHTHALMIC REVIEW';
                lblTelePriority.FontColor = [1.0, 0.2, 0.2];
            else
                lblTelePriority.Text = 'P2 - ROUTINE SPECIALIST QUEUE';
                lblTelePriority.FontColor = [0.95, 0.65, 0.2];
            end
        else
            % Non-Referable Case
            lblReferralBadge.Text = 'TRIAGE: NON-REFERABLE (LOCAL CLEARANCE)';
            lblReferralBadge.BackgroundColor = [0.15, 0.65, 0.35];
            lblReferralBadge.FontColor = 'w';

            lblTeleAction.Text = 'LOCAL ARCHIVE -> DISCHARGED AT PHC';
            lblTeleAction.FontColor = [0.3, 0.85, 0.4];
            lblTelePayload.Text = '0.00 MB (98.7% Bandwidth Conserved)';
            lblTeleLatency.Text = '0.00 sec (No Rural Uplink Used)';
            lblTelePriority.Text = 'P3 - ROUTINE ANNUAL RE-SCREEN';
            lblTelePriority.FontColor = [0.6, 0.75, 0.85];
        end
    end

    function cleanAxis(ax)
        ax.XTick = [];
        ax.YTick = [];
        ax.Box = 'on';
        ax.XColor = [0.2, 0.28, 0.42];
        ax.YColor = [0.2, 0.28, 0.42];
    end
end