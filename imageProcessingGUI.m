function create_gui()
  % 创建主窗口，使用 uifigure 替代 figure
    hFig = uifigure('Name', '图像处理工具', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);

    % 设置布局
    mainLayout = uigridlayout(hFig, [2, 1]);  % 分成两行一列
    mainLayout.RowHeight = {300, 300};        % 第一行300px，第二行100px

    % 图像显示区域
    imagePanel = uipanel('Parent', mainLayout, 'Title', '图像显示区', 'FontSize', 12);
    ax = axes('Parent', imagePanel, 'Position', [0, 0, 1, 1]);

    % 按钮面板
    buttonPanel = uipanel('Parent', mainLayout, 'Title', '操作按钮', 'FontSize', 12);
    buttonPanel.Layout.Column = 1;
    buttonPanel.Layout.Row = 2;

    % 按钮布局
    buttonLayout = uigridlayout(buttonPanel, [2, 4]); % 两行四列
    buttonLayout.RowHeight = {40, 40}; % 按钮的高度
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

    % 初始化handles
    handles = struct();
    guidata(hFig, handles);
    % 回调函数：加载图像
    function loadButton_Callback(hObject, eventdata)
        [fileName, pathName] = uigetfile({'*.jpg;*.png;*.bmp', '所有图像文件'});
        if fileName
            img = imread(fullfile(pathName, fileName));
            axes(handles.imageAxes);
            imshow(img);
            handles.img = img;  % 保存图像
            guidata(hFig, handles);
        end
    end

    % 回调函数：显示直方图
    function histButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);
            axes(handles.histAxes);
            imhist(img);  % 显示灰度直方图
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：直方图均衡化
    function equalizeButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            img = rgb2gray(handles.img);
            equalizedImg = histeq(img);  % 直方图均衡化
            axes(handles.imageAxes);
            imshow(equalizedImg);
            handles.enhancedImg = equalizedImg;
            guidata(hFig, handles);
        else
            errordlg('请先加载图像！', '错误');
        end
    end

    % 回调函数：直方图匹配
    function matchButton_Callback(hObject, eventdata)
        if isfield(handles, 'img')
            [fileName, pathName] = uigetfile({'*.jpg;*.png;*.bmp', '选择参考图像'});
            if fileName
                refImg = rgb2gray(imread(fullfile(pathName, fileName)));
                img = rgb2gray(handles.img);
                matchedImg = imhistmatch(img, refImg);  % 直方图匹配
                axes(handles.imageAxes);
                imshow(matchedImg);
                handles.enhancedImg = matchedImg;
                guidata(hFig, handles);
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
end
