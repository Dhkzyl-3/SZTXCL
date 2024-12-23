function create_gui()
    % 创建主窗口，使用 uifigure 替代 figure
    hFig = uifigure('Name', '图像处理工具', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);

    % 设置布局
    mainLayout = uigridlayout(hFig, [2, 1]);  % 分成两行一列
    mainLayout.RowHeight = {300, 300};        % 第一行300px，第二行100px

    % 图像显示区域
    imagePanel = uipanel('Parent', mainLayout, 'Title', '图像显示区', 'FontSize', 12);
    ax = axes('Parent', imagePanel, 'Position', [0, 0, 1, 1]);  % axes 填满整个 panel
    
    % 创建用于显示直方图的 axes
    histPanel = uipanel('Parent', mainLayout, 'Title', '直方图显示区', 'FontSize', 12);
    handles.histAxes = axes('Parent', histPanel, 'Position', [0, 0, 1, 1]);  % axes 填满整个 panel

    % 将 axes 存储在 handles 结构中，确保能够在回调中访问
    handles.imageAxes = ax;
    guidata(hFig, handles);  % 更新 handles

    % 按钮面板
    buttonPanel = uipanel('Parent', mainLayout, 'Title', '操作按钮', 'FontSize', 12);
    buttonPanel.Layout.Row = 2;

    % 按钮布局
    buttonLayout = uigridlayout(buttonPanel, [6, 4]); % 三行四列的布局
    buttonLayout.RowHeight = {40, 40, 40,40,40,40}; % 每个按钮的高度
    buttonLayout.ColumnWidth = {'1x', '1x', '1x', '1x'};  % 按钮均匀分布

    % 创建按钮
    loadButton = uibutton(buttonLayout, 'Text', '加载图像', 'ButtonPushedFcn', @(src, event) loadButton_Callback());
    histButton = uibutton(buttonLayout, 'Text', '显示直方图', 'ButtonPushedFcn', @(src, event) histButton_Callback());
    equalizeButton = uibutton(buttonLayout, 'Text', '直方图均衡化', 'ButtonPushedFcn', @(src, event) equalizeButton_Callback());
    matchButton = uibutton(buttonLayout, 'Text', '直方图匹配', 'ButtonPushedFcn', @(src, event) matchButton_Callback());

    linearButton = uibutton(buttonLayout, 'Text', '线性增强', 'ButtonPushedFcn', @(src, event) linearButton_Callback());
    grayButton = uibutton(buttonLayout, 'Text', '灰度转换', 'ButtonPushedFcn', @(src, event) grayButton_Callback());
    zoomButton = uibutton(buttonLayout, 'Text', '图像缩放', 'ButtonPushedFcn', @(src, event) zoomButton_Callback());
    rotateButton = uibutton(buttonLayout, 'Text', '图像旋转', 'ButtonPushedFcn', @(src, event) rotateButton_Callback());
    noiseButton = uibutton(buttonLayout, 'Text', '加噪声与滤波', 'ButtonPushedFcn', @(src, event) noiseButton_Callback());
% 新增边缘检测按钮
    robertButton = uibutton(buttonLayout, 'Text', 'Robert 边缘', 'ButtonPushedFcn', @(src, event) robertButton_Callback());
    prewittButton = uibutton(buttonLayout, 'Text', 'Prewitt 边缘', 'ButtonPushedFcn', @(src, event) prewittButton_Callback());
    sobelButton = uibutton(buttonLayout, 'Text', 'Sobel 边缘', 'ButtonPushedFcn', @(src, event) sobelButton_Callback());
    laplacianButton = uibutton(buttonLayout, 'Text', '拉普拉斯 边缘', 'ButtonPushedFcn', @(src, event) laplacianButton_Callback());
    extractTargetButton = uibutton(buttonLayout, 'Text', '特征提取', 'ButtonPushedFcn', @extractTargetButton_Callback);
    LBPButton = uibutton(buttonLayout,'Text', '提取 LBP 特征', 'ButtonPushedFcn', @(src, event) lbpButton_Callback(src,event));
    HOGButton = uibutton(buttonLayout, 'Text', '提取 HOG 特征', 'ButtonPushedFcn', @(src, event) hogButton_Callback(src,event));
    % 回调函数：加载图像
    function loadButton_Callback(hObject, eventdata)
        % 打开文件选择对话框，选择图像文件
        [fileName, pathName] = uigetfile({'*.jpg;*.png;*.bmp', '所有图像文件'});
        if fileName
            img = imread(fullfile(pathName, fileName));  % 读取选中的图像文件
            handles = guidata(hFig);  % 读取当前的 handles
            axes(handles.imageAxes);  % 将图像显示在指定的 axes 上
            imshow(img, 'Parent', handles.imageAxes);  % 在当前的 axes 中显示图像
            handles.img = img;  % 将图像保存到 handles 结构中
            guidata(hFig, handles);  % 更新 handles
        end
    end
    
    % 回调函数：显示直方图
    function histButton_Callback(hObject, eventdata)
        handles = guidata(hFig);  % 获取最新的 handles
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);  % 将图像转为灰度图
            axes(handles.histAxes);  % 切换到直方图的 axes
            imhist(img);  % 显示灰度直方图
        else
            errordlg('请先加载图像！', '错误');  % 如果没有加载图像，则弹出错误对话框
        end
    end
    % 回调函数：直方图均衡化
    function equalizeButton_Callback(hObject, eventdata)
        handles = guidata(hFig);  % 获取最新的 handles
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);  % 将图像转为灰度图
            eqImg = histeq(img);  % 进行直方图均衡化
            axes(handles.imageAxes);  % 显示均衡化后的图像
            imshow(eqImg, 'Parent', handles.imageAxes);  % 显示均衡化后的图像
            axes(handles.histAxes);  % 显示均衡化后的直方图
            imhist(eqImg);  % 显示均衡化后的直方图
        else
            errordlg('请先加载图像！', '错误');
        end
    end
    % 回调函数：直方图匹配
    function matchButton_Callback(hObject, eventdata)
        handles = guidata(hFig);  % 获取最新的 handles
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);  % 将图像转为灰度图

            % 目标图像（可以选择从文件加载一个目标图像，也可以使用默认图像）
            [fileName, pathName] = uigetfile({'*.jpg;*.png;*.bmp', '所有图像文件'}, '选择目标图像');
            if fileName
                targetImg = rgb2gray(imread(fullfile(pathName, fileName)));  % 读取目标图像并转换为灰度

                % 进行直方图匹配
                matchedImg = imhistmatch(img, targetImg);

                % 显示匹配后的图像
                axes(handles.imageAxes);
                imshow(matchedImg, 'Parent', handles.imageAxes);
                
                % 显示匹配后的直方图
                axes(handles.histAxes);
                imhist(matchedImg);
            else
                errordlg('未选择目标图像进行匹配', '错误');
            end
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：灰度转换
    function grayButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            grayImg = rgb2gray(handles.img);
            axes(handles.imageAxes);
            imshow(grayImg);
            handles.grayImg = grayImg;  % 保存灰度图
            guidata(hFig, handles);
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：线性对比度增强
    function linearButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);
            % 线性变换
            min_val = double(min(img(:)));
            max_val = double(max(img(:)));
            linearImg = uint8(255 * (double(img) - min_val) / (max_val - min_val));
            axes(handles.imageAxes);
            imshow(linearImg);
            handles.enhancedImg = linearImg;
            guidata(hFig, handles);
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：对数增强
    function logButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);
            % 对数变换
            c = 255 / log(1 + double(max(img(:))));
            logImg = uint8(c * log(1 + double(img)));
            axes(handles.imageAxes);
            imshow(logImg);
            handles.enhancedImg = logImg;
            guidata(hFig, handles);
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：指数增强
    function expButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);
            % 指数变换
            c = 255 / exp(double(max(img(:))) / 255);
            expImg = uint8(c * (exp(double(img) / 255) - 1));
            axes(handles.imageAxes);
            imshow(expImg);
            handles.enhancedImg = expImg;
            guidata(hFig, handles);
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：图像缩放
    function zoomButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            prompt = {'输入缩放因子：'};
            dlgtitle = '图像缩放';
            dims = [1 35];
            definput = {'1'};
            answer = inputdlg(prompt, dlgtitle, dims, definput);
            if ~isempty(answer)
                zoomFactor = str2double(answer{1});
                if ~isnan(zoomFactor)
                    % 缩放图像
                    resizedImg = imresize(handles.img, zoomFactor);
                    axes(handles.imageAxes);
                                        imshow(resizedImg);
                    handles.resizedImg = resizedImg;
                    guidata(hFig, handles);
                else
                    errordlg('请输入有效的缩放因子！', '错误');
                end
            end
        else
            errordlg('请先加载图像！', '错误');
        end
    end
    
    % 回调函数：图像旋转
    function rotateButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            prompt = {'输入旋转角度（度）：'};
            dlgtitle = '图像旋转';
            dims = [1 35];
            definput = {'0'};
            answer = inputdlg(prompt, dlgtitle, dims, definput);
            if ~isempty(answer)
                angle = str2double(answer{1});
                if ~isnan(angle)
                    % 旋转图像
                    rotatedImg = imrotate(handles.img, angle, 'bilinear', 'crop');
                    axes(handles.imageAxes);
                    imshow(rotatedImg);
                    handles.rotatedImg = rotatedImg;
                    guidata(hFig, handles);
                else
                    errordlg('请输入有效的旋转角度！', '错误');
                end
            end
        else
            errordlg('请先加载图像！', '错误');
        end
    end

 % 回调函数：加噪声和滤波
    function noiseButton_Callback()
        if isfield(handles, 'img')
            % 输入噪声类型和参数
            % 选择噪声类型对话框
            noiseType = listdlg('PromptString', '选择噪声类型', 'SelectionMode', 'single', ...
                'ListString', {'高斯噪声', '椒盐噪声'}, 'SelectionMode', 'single', 'OKString', '确认');
            if isempty(noiseType)
                return;
            end
            
            noiseLevel = inputdlg('请输入噪声强度（例如：0.05表示5%的噪声）：', '噪声强度', [1 50], {'0.05'});
            if isempty(noiseLevel)
                return;
            end
            noiseLevel = str2double(noiseLevel{1});
            if isnan(noiseLevel) || noiseLevel < 0 || noiseLevel > 1
                errordlg('噪声强度应在0到1之间！', '错误');
                return;
            end

            % 添加噪声
            switch noiseType
                case 1 % 高斯噪声
                    noisyImg = imnoise(handles.img, 'gaussian', 0, noiseLevel);
                case 2 % 椒盐噪声
                    noisyImg = imnoise(handles.img, 'salt & pepper', noiseLevel);
            end

            % 显示加噪后的图像
            axes(ax);
            imshow(noisyImg);
            title('加噪声后的图像');
            handles.noisyImg = noisyImg;
            guidata(hFig, handles);

            % 执行滤波处理
            % 空域滤波 - 均值滤波
            h = fspecial('average', [3 3]);  % 创建一个3x3均值滤波器
            filteredImg = imfilter(noisyImg, h);
            figure;
            imshow(filteredImg);
            title('均值滤波处理后的图像');

            % 频域滤波 - 理想低通滤波
            noisyImgGray = rgb2gray(noisyImg); % 转换为灰度图像
            [rows, cols] = size(noisyImgGray);
            f = fft2(double(noisyImgGray)); % 计算图像的傅里叶变换
            fshift = fftshift(f); % 将零频移到图像中心
            magnitude = abs(fshift); % 获取幅度谱
            phase = angle(fshift); % 获取相位谱

            % 创建理想低通滤波器
            D0 = 30;  % 截止频率
            [u, v] = meshgrid(-floor(cols/2):floor((cols-1)/2), -floor(rows/2):floor((rows-1)/2));
            D = sqrt(u.^2 + v.^2);  % 频率坐标的欧几里得距离
            H = double(D <= D0);  % 理想低通滤波器，D0为截止频率

            % 频域滤波
            fshiftFiltered = fshift .* H;  % 应用滤波器
            fFiltered = ifftshift(fshiftFiltered);  % 逆变换移回原位置
            filteredImgFreq = abs(ifft2(fFiltered));  % 逆傅里叶变换

            % 显示频域滤波后的图像
            figure;
            imshow(filteredImgFreq, []);
            title('频域滤波（理想低通滤波）后的图像');
        else
            errordlg('请先加载图像！', '错误');
        end
    end
 % 回调函数：Robert 边缘检测
    function robertButton_Callback(hObject, eventdata)
        handles = guidata(hFig);
    if isfield(handles, 'img')
        img = rgb2gray(handles.img);  % 转为灰度图
        
        % 定义 Robert 算子的两个卷积核
        Gx = [1 0; 0 -1];  % 水平方向的核
        Gy = [0 1; -1 0];  % 垂直方向的核
        
        % 使用 imfilter 进行卷积操作
        edgeX = imfilter(double(img), Gx, 'replicate');
        edgeY = imfilter(double(img), Gy, 'replicate');
        
        % 计算梯度幅值
        edgeImg = sqrt(edgeX.^2 + edgeY.^2);
        edgeImg = mat2gray(edgeImg);
        % 显示边缘检测结果
        axes(handles.imageAxes);
        imshow(edgeImg, []);
    else
        errordlg('请先加载图像！', '错误');
    end
end

    % 回调函数：Prewitt 边缘检测
    function prewittButton_Callback(hObject, eventdata)
    handles = guidata(hFig);
    if isfield(handles, 'img')
        img = rgb2gray(handles.img);  % 转为灰度图
        edgeImg = edge(img, 'Prewitt');  % 使用 Prewitt 算子进行边缘检测
        edgeImg = mat2gray(edgeImg);
        % 创建新窗口显示结果
        figure('Name', 'Prewitt 边缘检测', 'NumberTitle', 'off');
        imshow(edgeImg,[]);  % 显示边缘检测结果
    else
        errordlg('请先加载图像！', '错误');
    end
end
 % 回调函数：Sobel 边缘检测
    function sobelButton_Callback(hObject, eventdata)
    handles = guidata(hFig);
    if isfield(handles, 'img')
        img = rgb2gray(handles.img);  % 转为灰度图
        edgeImg = edge(img, 'Sobel');  % 使用 Sobel 算子进行边缘检测
        edgeImg = mat2gray(edgeImg);
        % 创建新窗口显示结果
        figure('Name', 'Sobel 边缘检测', 'NumberTitle', 'off');
        imshow(edgeImg,[]);  % 显示边缘检测结果
    else
        errordlg('请先加载图像！', '错误');
    end
end

    % 回调函数：拉普拉斯边缘检测
      function laplacianButton_Callback(hObject, eventdata)
      handles = guidata(hFig);
      if isfield(handles, 'img')
          img = rgb2gray(handles.img);  % 转为灰度图

        % 使用拉普拉斯算子进行边缘检测
          edgeImg = edge(img, 'log');  % 拉普拉斯算子

        % 增强对比度
          edgeImg = mat2gray(edgeImg);

        % 创建新窗口显示结果
          figure('Name', '拉普拉斯边缘检测', 'NumberTitle', 'off');
          imshow(edgeImg,[]);  % 显示边缘检测结果
      else
          errordlg('请先加载图像！', '错误');
      end
      end
function extractTargetButton_Callback(src, event)
    handles = guidata(src);  % 获取句柄
    if isfield(handles, 'img')
        % 1. 读取图像并转化为灰度图
        img = handles.img;  % 获取图像数据
        grayImg = rgb2gray(img);  % 转换为灰度图
        % 2. 高斯模糊，去噪
        blurredImg = imgaussfilt(grayImg, 2);  % 高斯滤波，去除噪声
        % 3. 边缘检测
        edgeImg = edge(blurredImg, 'Sobel');  % Sobel 算子边缘检测
        % 4. 形态学操作，增强边缘
        se = strel('disk', 2);  % 创建一个大小为2的圆形结构元素
        dilatedImg = imdilate(edgeImg, se);  % 膨胀操作
        erodedImg = imerode(dilatedImg, se);  % 腐蚀操作
        % 5. 连通域分析，提取目标
        stats = regionprops(erodedImg, 'BoundingBox', 'Area', 'Centroid');
        % 6. 过滤掉小区域
        minArea = 500;  % 设置最小区域面积，过滤掉噪声
        filteredStats = stats([stats.Area] > minArea);  % 过滤小区域
        % 7. 绘制检测到的目标
        figure('Name', '目标提取结果', 'NumberTitle', 'off');
        imshow(img);  % 显示原图
        hold on;
        for k = 1:length(filteredStats)
            % 绘制矩形框标出检测到的目标
            rectangle('Position', filteredStats(k).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
            % 绘制目标的质心
            plot(filteredStats(k).Centroid(1), filteredStats(k).Centroid(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);           
            % 提取目标区域
            targetRegion = imcrop(img, filteredStats(k).BoundingBox); 
        end
            for k = 1:length(filteredStats)
            % 绘制矩形框标出检测到的目标
            rectangle('Position', filteredStats(k).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
            % 绘制目标的质心
            plot(filteredStats(k).Centroid(1), filteredStats(k).Centroid(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);           
            % 提取目标区域
            targetRegion = imcrop(img, filteredStats(k).BoundingBox);
            % 对目标区域进行 LBP 特征提取
            lbpFeature(targetRegion);            
            % 对目标区域进行 HOG 特征提取
            hogFeature(targetRegion);
            
            end
            hold off;
            title('目标提取结果');
    end
end

% LBP特征提取与显示
function lbpFeature(region)
    % 转换为灰度图像
    if size(region, 3) == 3
        regionGray = rgb2gray(region);
    else
        regionGray = region;
    end
    
    [N, M] = size(regionGray);  % 获取图像的尺寸
    
    lbp = zeros(N, M);  % 初始化 LBP 特征图
    
    % 计算 LBP 特征
    for i = 2:N-1
        for j = 2:M-1
            % 邻域像素位置
            neighbor = [i-1, j-1; i-1, j; i-1, j+1;i, j+1; i+1, j+1;i+1, j; i+1, j-1;i, j-1]; 
            count = 0;
            for k = 1:8
                if regionGray(neighbor(k, 1), neighbor(k, 2)) > regionGray(i, j)
                    count = count + 2^(8 - k);  % LBP 二进制编码
                end
            end
            lbp(i, j) = count;  % 设置 LBP 特征值
        end
    end
    
    lbp = uint8(lbp);  % 将 LBP 特征图转为 8 位无符号整数
    
    % 显示 LBP 特征图
    figure;
    imshow(lbp, []);
    title('目标区域 LBP 特征图');
end

% HOG特征提取与显示
function hogFeature(region)
    % 转换为灰度图像
    if size(region, 3) == 3
        grayRegion = rgb2gray(region);
    else
        grayRegion = region;
    end
    
    % 计算 HOG 特征
    [hogFeatures, visualization] = extractHOGFeatures(grayRegion, 'CellSize', [8 8], 'BlockSize', [2 2]);
    
    % 显示 HOG 特征的可视化
    figure;
    imshow(grayRegion);  % 显示灰度图像
    hold on;
    plot(visualization);
    title('目标区域 HOG 特征可视化');
end

function lbpButton_Callback(src, event)
    % 获取 GUI 数据
    handles = guidata(src);
    
    % 获取显示图像的句柄
    img = getimage(handles.imageAxes);
    
    % 检查图像是否为空
    if isempty(img)
        msgbox('请先加载图像。', '错误', 'error');
        return;
    end
    
    % 如果是彩色图像，转换为灰度图像
    if size(img, 3) == 3
        image = rgb2gray(img);  % 转换为灰度图像
    else
        image = img;  % 如果已经是灰度图像，直接使用
    end
    
    [N, M] = size(image);  % 获取图像的尺寸
    
    lbp = zeros(N, M);  % 初始化 LBP 特征图
    
    % 计算 LBP 特征
    for i = 2:N-1
        for j = 2:M-1
            % 邻域像素位置
            neighbor = [i-1, j-1; i-1, j; i-1, j+1; i, j+1;i+1, j+1;i+1, j;i+1, j-1;i, j-1];
            
            count = 0;
            for k = 1:8
                if image(neighbor(k, 1), neighbor(k, 2)) > image(i, j)
                    count = count + 2^(8 - k);  % LBP 二进制编码
                end
            end
            lbp(i, j) = count;  % 设置 LBP 特征值
        end
    end
    
    lbp = uint8(lbp);  % 将 LBP 特征图转为 8 位无符号整数
    
    % 显示 LBP 特征图
    figure;
    imshow(lbp, []);
    title('LBP 特征图');
    
    % 获取一个 8x8 子区域，并计算其直方图
    subim = lbp(1:8, 1:8);
    figure;
    imhist(subim);  % 显示直方图
    title('第一个子区域直方图');
end

function hogButton_Callback(src, event)
    % 提取 HOG 特征的回调函数
    handles = guidata(src);  % 获取 GUI 数据
    img = getimage(handles.imageAxes);  % 获取图像

    if isempty(img)
        msgbox('请先加载图像。', '错误', 'error');
        return;
    end
    
    % 如果是彩色图像，转换为灰度图像
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % 计算 HOG 特征
    [hogFeatures, visualization] = extractHOGFeatures(grayImg, 'CellSize', [8 8], 'BlockSize', [2 2]);

    
    % 显示 HOG 特
    figure;
    imshow(grayImg);  % 显示灰度图像
    hold on;
    plot(visualization);
    title('HOG 特征可视化');
end
end