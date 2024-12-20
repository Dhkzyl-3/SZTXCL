classdef ImageProcessingApp2 < matlab.apps.AppBase
    % 该应用程序的属性
    properties (Access = public)
        UIFigure             matlab.ui.Figure
        UIAxes               matlab.ui.control.UIAxes % 使用 UIAxes 类型
        LoadImageButton      matlab.ui.control.Button
        GrayHistButton       matlab.ui.control.Button
        EqualizeButton       matlab.ui.control.Button
        ContrastButton       matlab.ui.control.Button
        ScaleButton          matlab.ui.control.Button
        RotateButton         matlab.ui.control.Button
        NoiseButton          matlab.ui.control.Button
        EdgeButton           matlab.ui.control.Button
        ExtractButton        matlab.ui.control.Button
        FeatureButton        matlab.ui.control.Button
    end
    
    properties (Access = private)
        Image % 原始图像
    end
    
    methods (Access = private)
        
        % 加载并显示图像
        function loadImage(app)
            [file, path] = uigetfile('*.*', '选择图像');
            if file
                img = imread(fullfile(path, file));
                app.Image = rgb2gray(img);  % 转化为灰度图像
                imshow(app.Image, 'Parent', app.UIAxes);
            end
        end
        
        % 显示灰度直方图
        function showGrayHist(app)
            figure; imhist(app.Image); title('Gray Histogram');
        end
        
        % 直方图均衡化
        function equalizeHistogram(app)
            eq_img = histeq(app.Image);
            imshow(eq_img, 'Parent', app.UIAxes);
        end
        
        % 对比度增强（线性变换、对数变换、指数变换）
        function contrastEnhance(app, transformType)
            if transformType == "linear"
                contrast_img = imadjust(app.Image);
            elseif transformType == "log"
                c = 1;
                contrast_img = uint8(c * log(double(app.Image) + 1));
            elseif transformType == "exp"
                c = 1;
                contrast_img = uint8(c * (double(app.Image) .^ 2));
            end
            imshow(contrast_img, 'Parent', app.UIAxes);
        end
        
        % 图像缩放
        function scaleImage(app, scaleFactor)
            scaled_img = imresize(app.Image, scaleFactor);
            imshow(scaled_img, 'Parent', app.UIAxes);
        end
        
        % 图像旋转
        function rotateImage(app, angle)
            rotated_img = imrotate(app.Image, angle);
            imshow(rotated_img, 'Parent', app.UIAxes);
        end
        
        % 添加噪声
        function addNoise(app, noiseType, noiseLevel)
            noisy_img = imnoise(app.Image, noiseType, noiseLevel);
            imshow(noisy_img, 'Parent', app.UIAxes);
            
            % 空域域滤波：使用中值滤波
            if noiseType == "salt & pepper"
                filtered_img = medfilt2(noisy_img);
                figure; imshow(filtered_img); title('Median Filtered Image');
            end
            
            % 频域滤波：使用高通滤波器
            noisy_fft = fft2(noisy_img);
            noisy_fft_shifted = fftshift(noisy_fft);
            [M, N] = size(noisy_img);
            X = 1:N; Y = 1:M;
            [X, Y] = meshgrid(X, Y);
            D0 = 30; % 截止频率
            H = 1 - exp(-((X - N/2).^2 + (Y - M/2).^2) / (2 * D0^2));
            filtered_fft = noisy_fft_shifted .* H;
            filtered_img = real(ifft2(ifftshift(filtered_fft)));
            figure; imshow(filtered_img, []); title('Frequency Domain Filtered Image');
        end
        
        % 边缘提取
        function edgeDetection(app, operatorType)
            if operatorType == "robert"
                edge_img = edge(app.Image, 'Roberts');
            elseif operatorType == "prewitt"
                edge_img = edge(app.Image, 'Prewitt');
            elseif operatorType == "sobel"
                edge_img = edge(app.Image, 'Sobel');
            elseif operatorType == "laplacian"
                edge_img = edge(app.Image, 'log');
            end
            imshow(edge_img, 'Parent', app.UIAxes);
        end
        
        % 目标提取（基于阈值分割）
        function extractObject(app)
            threshold = graythresh(app.Image);
            binary_img = imbinarize(app.Image, threshold);
            imshow(binary_img, 'Parent', app.UIAxes);
        end
        
        % 特征提取（LBP和HOG）
        function extractFeatures(app, featureType)
            if featureType == "LBP"
                lbp_features = extractLBPFeatures(app.Image);
                disp('LBP Features:');
                disp(lbp_features);
            elseif featureType == "HOG"
                [hog_features, visualization] = extractHOGFeatures(app.Image);
                figure; plot(visualization); title('HOG Features');
            end
        end
    end
    
    methods (Access = public)
        
        % 构造函数
        function app = ImageProcessingApp2()
            % 创建界面并设置其属性
            createUI(app);
        end
        
        % 创建界面
        function createUI(app)
            % 创建应用程序窗口
            app.UIFigure = uifigure('Position', [100, 100, 800, 600], 'Name', 'Image Processing App');
            
            % 创建Axes并设置其位置和大小
            app.UIAxes = uiaxes(app.UIFigure); % 正确的UIAxes创建方式
            app.UIAxes.Position = [0.3, 0.3, 0.65, 0.65]; % 修改为正确的坐标设置
            
            % 创建按钮
            app.LoadImageButton = uibutton(app.UIFigure, 'push', 'Text', '加载图像', ...
                'Position', [20, 540, 100, 30], 'ButtonPushedFcn', @(btn, event) loadImage(app));
            app.GrayHistButton = uibutton(app.UIFigure, 'push', 'Text', '显示灰度直方图', ...
                'Position', [20, 490, 120, 30], 'ButtonPushedFcn', @(btn, event) showGrayHist(app));
            app.EqualizeButton = uibutton(app.UIFigure, 'push', 'Text', '直方图均衡化', ...
                'Position', [20, 440, 120, 30], 'ButtonPushedFcn', @(btn, event) equalizeHistogram(app));
            app.ContrastButton = uibutton(app.UIFigure, 'push', 'Text', '对比度增强', ...
                'Position', [20, 390, 120, 30], 'ButtonPushedFcn', @(btn, event) contrastEnhance(app, 'linear'));
            app.ScaleButton = uibutton(app.UIFigure, 'push', 'Text', '缩放图像', ...
                'Position', [20, 340, 120, 30], 'ButtonPushedFcn', @(btn, event) scaleImage(app, 1.5));
            app.RotateButton = uibutton(app.UIFigure, 'push', 'Text', '旋转图像', ...
                'Position', [20, 290, 120, 30], 'ButtonPushedFcn', @(btn, event) rotateImage(app, 45));
            app.NoiseButton = uibutton(app.UIFigure, 'push', 'Text', '添加噪声', ...
                'Position', [20, 240, 120, 30], 'ButtonPushedFcn', @(btn, event) addNoise(app, 'salt & pepper', 0.02));
            app.EdgeButton = uibutton(app.UIFigure, 'push', 'Text', '边缘提取', ...
                'Position', [20, 190, 120, 30], 'ButtonPushedFcn', @(btn, event) edgeDetection(app, 'sobel'));
            app.ExtractButton = uibutton(app.UIFigure, 'push', 'Text', '目标提取', ...
                'Position', [20, 140, 120, 30], 'ButtonPushedFcn', @(btn, event) extractObject(app));
            app.FeatureButton = uibutton(app.UIFigure, 'push', 'Text', '特征提取', ...
                'Position', [20, 90, 120, 30], 'ButtonPushedFcn', @(btn, event) extractFeatures(app, 'HOG'));
        end
    end
end

