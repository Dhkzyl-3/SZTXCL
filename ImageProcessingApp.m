function ImageProcessingApp()
    % 创建一个简单的 GUI 界面
    fig = figure('Name', 'Image Processing App', ...
                'NumberTitle', 'off', ...
                'Position', [100, 100, 960, 650], ...
                'CloseRequestFcn', @closeFig); % 关闭回调函数

    % 显示原始图像、处理后的图像以及直方图的区域
    ax1 = axes('Parent', fig, 'Position', [0.05, 0.4, 0.4, 0.55]);
    ax2 = axes('Parent', fig, 'Position', [0.55, 0.4, 0.4, 0.55]);
    ax3 = axes('Parent', fig, 'Position', [0.05, 0.05, 0.4, 0.3], 'Visible', 'off'); % 用于显示直方图

    % 按钮和控件
    uicontrol('Style', 'pushbutton', 'String', 'Open Image', ...
              'Position', [50, 40, 100, 30], 'Callback', @openImage);
    uicontrol('Style', 'pushbutton', 'String', 'Convert to Grayscale', ...
              'Position', [160, 40, 150, 30], 'Callback', @convertToGray);
    uicontrol('Style', 'pushbutton', 'String', 'Histogram Equalization', ...
              'Position', [270, 40, 150, 30], 'Callback', @histeqIzation);
    uicontrol('Style', 'pushbutton', 'String', 'Histogram Matching', ...
              'Position', [430, 40, 150, 30], 'Callback', @histMatching);
    uicontrol('Style', 'pushbutton', 'String', 'Linear Contrast Stretch', ...
              'Position', [580, 40, 150, 30], 'Callback', @linearContrastStretch);
    uicontrol('Style', 'pushbutton', 'String', 'Logarithmic Transformation', ...
              'Position', [730, 40, 150, 30], 'Callback', @logTransformation);
    uicontrol('Style', 'pushbutton', 'String', 'Exponential Transformation', ...
              'Position', [880, 40, 150, 30], 'Callback', @expTransformation);
    uicontrol('Style', 'pushbutton', 'String', 'Scale Image', ...
              'Position', [50, 80, 100, 30], 'Callback', @scaleImage);
    uicontrol('Style', 'pushbutton', 'String', 'Rotate Image', ...
              'Position', [160, 80, 150, 30], 'Callback', @rotateImage);

    % 按钮回调函数 - 打开图像
    function openImage(~, ~)
        [fileName, pathName] = uitgetfile('*.jpg;*.png;*.bmp;*.tif;*.*;.jpeg;*.JPG', ...
            'Select an image file (*.jpg;*.png;*.bmp;*.tif;*.*;.jpeg;*.JPG)');
        if isequal(fileName, 0)
            return;
        end
        img = imread(fullfile(pathName, fileName));
        setappdata(fig, 'img', img); % 保存图像数据
        axes(ax1); imshow(img); title('Original Image');
        plotHistogram(img); % 显示图像的灰度直方图
    end

    % 按钮回调函数 - 转换为灰度图像
    function convertToGray(~, ~)
        img = getappdata(fig, 'img');
        if isempty(img)
            msgbox('Please open an image first.');
            return;
        end
        grayImg = rgb2gray(img);
        axes(ax2); imshow(grayImg); title('Grayscale Image');
        setappdata(fig, 'grayImg', grayImg); % 保存灰度图像数据
        plotHistogram(grayImg); % 显示灰度图像的直方图
    end

    % 按钮回调函数 - 直方图均衡化
    function histeqIzation(~, ~)
        grayImg = getappdata(fig, 'grayImg');
        if isempty(grayImg)
            msgbox('Please convert to grayscale first.');
            return;
        end
        eqImg = histogramEqualization(grayImg);
        axes(ax2); imshow(eqImg); title('Histogram Equalized Image');
        setappdata(fig, 'eqImg', eqImg); % 保存均衡化后的图像数据
        plotHistogram(eqImg); % 显示均衡化后的直方图
    end

    % 按钮回调函数 - 直方图匹配
    function histMatching(~, ~)
        grayImg = getappdata(fig, 'grayImg');
        if isempty(grayImg)
            msgbox('Please convert to grayscale first.');
            return;
        end
        % 使用一个目标直方图来进行匹配（例如均匀分布直方图）
        targetHist = uint8(linspace(0, 255, 256)); % 目标直方图可以是均匀分布直方图
        matchedImg = histogramMatching(grayImg, targetHist);
        axes(ax2); imshow(matchedImg); title('Histogram Matched Image');
        setappdata(fig, 'matchedImg', matchedImg); % 保存匹配后的图像数据)
        plotHistogram(matchedImg); % 显示匹配后的直方图
    end

    % 按钮回调函数 - 线性对比度拉伸
    function linearContrastStretch(~, ~)
        grayImg = getappdata(fig, 'grayImg');
        if isempty(grayImg)
            msgbox('Please convert to grayscale first.');
            return;
        end
        stretchedImg = linearStretch(grayImg);
        axes(ax2); imshow(stretchedImg); title('Linear Contrast Stretched Image');
        setappdata(fig, 'stretchedImg', stretchedImg); % 保存拉伸后的图像数据)
        plotHistogram(stretchedImg); % 显示拉伸后的直方图
    end

    % 按钮回调函数 - 对数变换
    function logTransformation(~, ~)
        grayImg = getappdata(fig, 'grayImg');
        if isempty(grayImg)
            msgbox('Please convert to grayscale first.');
            return;
        end
        logImg = logTrans(grayImg);
        axes(ax2); imshow(logImg); title('Logarithmic Transformed Image');
        setappdata(fig, 'logImg', logImg); % 保存对数变换后的图像数据)
        plotHistogram(logImg); % 显示对数变换后的直方图
    end

    % 按钮回调函数 - 指数变换
    function expTransformation(~, ~)
        grayImg = getappdata(fig, 'grayImg');
        if isempty(grayImg)
            msgbox('Please convert to grayscale first.');
            return;
        end
        expImg = expTrans(grayImg);
        axes(ax2); imshow(expImg); title('Exponential Transformed Image');
        setappdata(fig, 'expImg', expImg); % 保存指数变换后的图像数据)
        plotHistogram(expImg); % 显示指数变换后的直方图
    end

    % 按钮回调函数 - 缩放图像
    function scaleImage(~, ~)
        img = getappdata(fig, 'img');
        if isempty(img)
            msgbox('Please open an image first.');
            return;
        end
        [scaleX, scaleY] = inputdlg('Enter scaling factors for X and Y:');
        scaleX = str2double(scaleX);
        scaleY = str2double(scaleY);
        imgScaled = imresize(img, round(sprintf('%f %f', scaleX * size(img, 1), scaleY * size(img, 2))));
        axes(ax1); imshow(imgScaled); title('Scaled Image');
    end

    % 按钮回调函数 - 旋转图像
    function rotateImage(~, ~)
        img = getappdata(fig, 'img');
        if isempty(img)
            msgbox('Please open an image first.');
            return;
        end
        angle = inputdlg('Enter rotation angle in degrees:');
        angle = str2double(angle);
        imgRotated = imrotate(img, angle);
        axes(ax1); imshow(imgRotated); title('Rotated Image');
    end
    % 绘制直方图的函数
    function plotHistogram(img)
        % 检查图像是否为RGB，如果是则转换为灰度图像
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        % 计算并绘制直方图
        [counts, binLocations] = imhist(img);
        bars(binLocations, counts);
        title('Histogram');
        xlim([0 255]); % 灰度范围
        ylim([0 max(counts) + 10]); % 设置y轴范围
    end
    % 直方图均衡化函数
    function eqImg = histogramEqualization(grayImg)
        % 计算累积分布函数 (CDF)
        cdf =cumsum(imhist(grayImg));
        cdf = cdf / numel(grayImg); % 归一化到 [0, 1]
        cdf = uint8(cdf * 256); % 归一化到 [0, 255]
        % 使用 CDF 进行直方图均衡化
        eqImg = uint8(cdf(double(grayImg) + 1)); % double转换确保索引为整数类型，+1确保索引从1开始，而不是0。
    end
    % 直方图匹配函数
    function matchedImg = histogramMatching(inputImg, targetHist)
        % 计算输入图像的累积分布函数 (CDF)
        cdfIn =cumsum(imhist(inputImg));
        % 计算目标直方图的累积分布函数 (CDF)
        cdfTarget =cumsum(targetHist);
        cdfTarget = cdfTarget / numel(targetHist); % 归一化到 [0, 1]
        cdfTarget = uint8(cdfTarget * 256); % 归一化到 [0, 255]
        % 为每个输入像素值分配目标CDF中的新值以进行匹配
        matchedImg = zeros(size(inputImg), 'uint8'); % 初始化输出图像矩阵，大小与输入相同，类型为uint8。
        for i = 1:numel(inputImg)
            idx = find(cdfTarget >= cdfIn(i), 1); % 找到最接近的新值索引，使得目标CDF值大于等于当前输入CDF值。
            matchedImg(i) = uint8(idx - 1); % 根据找到的索引分配新值，减1是因为MATLAB索引从1开始，而CDF索引从0开始。
        end
    end
    % 线性对比度拉伸函数
    function stretchedImg = linearStretch(grayImg)
        minVal = min(grayImg(:));
        maxVal = max(grayImg(:));
        stretchedImg = uint8((double(grayImg) - minVal) / (maxVal - minVal) * 255); % 确保结果在 [0,255] 范围内。
    end
    % 对数变换函数
    function logImg = logTrans(grayImg)
        c = 255 / log(1 + max(grayImg(:))); % 常数因子，确保结果在 [0, 255] 范围内。
        logImg = uint8(c * log(double(grayImg) + 1)); % 加1以避免对数零问题。
    end
  % 指数变换函数
function transformedImg = expTrans(grayImg)
    c = (exp(max(grayImg(:))) - 1) / 255; % 常数因子，确保结果在 [0, 255] 范围内。
    transformedImg = c * (exp(double(grayImg)) - 1); % 减1以避免指数零问题。
end

end
