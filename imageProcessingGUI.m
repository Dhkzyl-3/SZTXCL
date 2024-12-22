function varargout = imageProcessingGUI(varargin)

    % 创建一个新的图形窗口
    hFig = figure('Name', '图像处理工具', 'NumberTitle', 'off', 'Position', [100, 100, 600, 500]);

    % 创建按钮：加载图像
    loadButton = uicontrol('Style', 'pushbutton', 'String', '加载图像', 'Position', [20, 450, 100, 30], 'Callback', @loadImageButton_Callback);

    % 创建按钮：转换为灰度图
    convertButton = uicontrol('Style', 'pushbutton', 'String', '转换为灰度', 'Position', [140, 450, 100, 30], 'Callback', @convertToGrayButton_Callback);

    % 创建按钮：保存图像
    saveButton = uicontrol('Style', 'pushbutton', 'String', '保存图像', 'Position', [260, 450, 100, 30], 'Callback', @saveImageButton_Callback);

    % 创建按钮：线性对比度变换
    linearButton = uicontrol('Style', 'pushbutton', 'String', '线性对比度变换', 'Position', [380, 450, 130, 30], 'Callback', @linearContrastButton_Callback);

    % 创建按钮：对数对比度变换
    logButton = uicontrol('Style', 'pushbutton', 'String', '对数对比度变换', 'Position', [20, 400, 130, 30], 'Callback', @logContrastButton_Callback);

    % 创建按钮：指数对比度变换
    expButton = uicontrol('Style', 'pushbutton', 'String', '指数对比度变换', 'Position', [160, 400, 130, 30], 'Callback', @expContrastButton_Callback);

    % 创建显示区域
    handles.imageAxes = axes('Parent', hFig, 'Position', [0.1, 0.1, 0.8, 0.7]);

    % 初始化 handles 结构
    handles.output = hFig;
    handles.img = [];
    handles.grayImg = [];
    handles.enhancedImg = [];  % 存储增强后的图像

    % 更新 handles 结构
    guidata(hFig, handles);

    % 设置输出
    if nargout
        varargout{1} = hFig;
    end

    % 按钮回调函数：加载图像
    function loadImageButton_Callback(hObject, eventdata)
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'}, '选择一张图像');
        if filename ~= 0
            img = imread(fullfile(pathname, filename));
            axes(handles.imageAxes);
            imshow(img);
            handles.img = img;
            set(convertButton, 'Enable', 'on');  % 启用转换按钮
            guidata(hFig, handles);
        else
            errordlg('未选择图像文件！', '错误');
        end
    end

    % 按钮回调函数：转换图像为灰度图
    function convertToGrayButton_Callback(hObject, eventdata)
        if ~isempty(handles.img)
            grayImg = rgb2gray(handles.img);
            axes(handles.imageAxes);
            imshow(grayImg);
            handles.grayImg = grayImg;
            set(linearButton, 'Enable', 'on');  % 启用对比度变换按钮
            set(logButton, 'Enable', 'on');
            set(expButton, 'Enable', 'on');
            guidata(hFig, handles);
        else
            errordlg('未加载任何图像！', '错误');
        end
    end

    % 按钮回调函数：保存图像
    function saveImageButton_Callback(hObject, eventdata)
        if ~isempty(handles.enhancedImg)
            [filename, pathname] = uiputfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'}, '保存图像为');
            if filename ~= 0
                imwrite(handles.enhancedImg, fullfile(pathname, filename));
            end
        else
            errordlg('没有图像可以保存！', '错误');
        end
    end

    % 线性对比度增强函数
    function linearContrastButton_Callback(hObject, eventdata)
        if ~isempty(handles.grayImg)
            % 线性对比度增强：简单的线性变换 f(x) = a*x + b
            img = double(handles.grayImg);  % 转换为 double 类型进行处理
            minVal = min(img(:));
            maxVal = max(img(:));
            % 线性变换公式： (x - min) / (max - min) * 255
            enhancedImg = (img - minVal) / (maxVal - minVal) * 255;
            enhancedImg = uint8(enhancedImg);  % 转换回 uint8 类型
            axes(handles.imageAxes);
            imshow(enhancedImg);
            handles.enhancedImg = enhancedImg;
            guidata(hFig, handles);
        else
            errordlg('请先转换为灰度图！', '错误');
        end
    end

    % 对数对比度增强函数
    function logContrastButton_Callback(hObject, eventdata)
        if ~isempty(handles.grayImg)
            % 对数对比度增强：f(x) = c * log(1 + x)
            img = double(handles.grayImg);  % 转换为 double 类型
            c = 255 / log(1 + max(img(:)));  % 计算常数 c，确保输出范围为 0 到 255
            enhancedImg = c * log(1 + img);
            enhancedImg = uint8(enhancedImg);  % 转换回 uint8 类型
            axes(handles.imageAxes);
            imshow(enhancedImg);
            handles.enhancedImg = enhancedImg;
            guidata(hFig, handles);
        else
            errordlg('请先转换为灰度图！', '错误');
        end
    end

    % 指数对比度增强函数
    function expContrastButton_Callback(hObject, eventdata)
        if ~isempty(handles.grayImg)
            % 指数对比度增强：f(x) = c * (e^(x / b) - 1)
            img = double(handles.grayImg);  % 转换为 double 类型
            b = 50;  % 控制指数增长的参数
            c = 255 / (exp(max(img(:)) / b) - 1);  % 计算常数 c，确保输出范围为 0 到 255
            enhancedImg = c * (exp(img / b) - 1);
            enhancedImg = uint8(enhancedImg);  % 转换回 uint8 类型
            axes(handles.imageAxes);
            imshow(enhancedImg);
            handles.enhancedImg = enhancedImg;
            guidata(hFig, handles);
        else
            errordlg('请先转换为灰度图！', '错误');
        end
    end

end
