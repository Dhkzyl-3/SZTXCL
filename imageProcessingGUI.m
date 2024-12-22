function varargout = imageProcessingGUI(varargin)

   % 创建一个新的图形窗口
    hFig = figure('Name', '图像处理工具', 'NumberTitle', 'off', 'Position', [100, 100, 600, 400]);

    % 创建一个按钮，加载图像
    uicontrol('Style', 'pushbutton', 'String', '加载图像', 'Position', [20, 350, 100, 30], 'Callback', @loadImageButton_Callback);

    % 创建一个按钮，转换图像为灰度图
    uicontrol('Style', 'pushbutton', 'String', '转换为灰度', 'Position', [140, 350, 100, 30], 'Callback', @convertToGrayButton_Callback);

    % 创建一个按钮，保存图像
    uicontrol('Style', 'pushbutton', 'String', '保存图像', 'Position', [260, 350, 100, 30], 'Callback', @saveImageButton_Callback);

    % 创建一个axes，用于显示图像
    handles.imageAxes = axes('Parent', hFig, 'Position', [0.1, 0.1, 0.8, 0.7]);

    % 初始化 handles 结构
    handles.output = hFig;
    handles.img = [];      % 用于存储加载的原始图像
    handles.grayImg = [];  % 用于存储灰度图像

    % 更新 handles 结构
    guidata(hFig, handles);

    % 设置输出
    if nargout
        varargout{1} = hFig;
    end
    % 按钮回调函数：加载图像
    function loadImageButton_Callback(hObject, eventdata)
        % 打开文件对话框，选择要加载的图像
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'}, '选择一张图像');
        
        if filename ~= 0
            % 加载图像，并显示在指定的axes中
            img = imread(fullfile(pathname, filename));
            axes(handles.imageAxes);  % 设置要显示图像的axes
            imshow(img);  % 显示图像
            % 将图像数据保存到handles结构中
            handles.img = img;
            guidata(hFig, handles);  % 更新handles结构
        end
    end
    % 按钮回调函数：转换图像为灰度图
    function convertToGrayButton_Callback(hObject, eventdata)
        if ~isempty(handles.img)
            % 将加载的彩色图像转换为灰度图
            grayImg = rgb2gray(handles.img);
            % 显示灰度图像
            axes(handles.imageAxes);
            imshow(grayImg);
            % 将灰度图像保存到handles结构中
            handles.grayImg = grayImg;
            guidata(hFig, handles);  % 更新handles结构
        else
            % 如果没有加载图像，弹出错误对话框
            errordlg('未加载任何图像！', '错误');
        end
    end
    % 按钮回调函数：保存图像
    function saveImageButton_Callback(hObject, eventdata)
        if ~isempty(handles.grayImg)
            % 打开保存文件对话框，选择保存路径和文件名
            [filename, pathname] = uiputfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'}, '保存图像为');
            if filename ~= 0
                % 将当前的灰度图像保存到选定的文件
                imwrite(handles.grayImg, fullfile(pathname, filename));
            end
        else
            % 如果没有灰度图像可保存，弹出错误对话框
            errordlg('没有图像可以保存！', '错误');
        end
    end
end

